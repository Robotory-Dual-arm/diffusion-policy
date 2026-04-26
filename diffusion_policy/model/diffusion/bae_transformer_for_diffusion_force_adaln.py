import copy
import math
from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)

def build_sinusoidal_pos_emb(length: int, dim: int) -> torch.Tensor:
    pe = torch.zeros(length, dim)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)
    )
    pos_term = position * div_term
    pe[:, 0::2] = torch.sin(pos_term)
    pe[:, 1::2] = torch.cos(pos_term)[:, :pe[:, 1::2].shape[1]]
    return pe.unsqueeze(0)

def print_attn_mask_grid(mask: torch.Tensor, name: str = "mask"):
    """
    mask: (T,S) 또는 (T,T), 값이 0.0(허용) / -inf(차단) 형태라고 가정
    """
    if mask is None:
        print(f"{name}: None")
        return

    m = mask.detach().cpu()
    allowed = torch.isfinite(m) & (m == 0)   # 허용

    T, S = m.shape
    print(f"\n{name} shape = ({T}, {S})")
    print("    " + " ".join([f"{j:2d}" for j in range(S)]))
    for i in range(T):
        row = []
        for j in range(S):
            row.append(" O" if allowed[i, j] else " X")
        print(f"{i:2d}: " + "".join(row))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_activation(activation: str) -> nn.Module:
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU(approximate="tanh")
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def zero_linear_chunk(linear: nn.Linear, chunk_index: int, num_chunks: int):
    chunk_size = linear.out_features // num_chunks
    start = chunk_index * chunk_size
    end = start + chunk_size
    torch.nn.init.zeros_(linear.weight[start:end])
    if linear.bias is not None:
        torch.nn.init.zeros_(linear.bias[start:end])


class ConditionSelfAttnEncoderLayer(nn.Module):
    """
    DiT-policy reference encoder에 맞춘 condition self-attention block.
    pos는 Q/K에만 더하고, value/residual에는 원래 condition token을 유지한다.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation(activation)

    def forward(self, src: torch.Tensor, pos: Optional[torch.Tensor] = None):
        q = k = src if pos is None else src + pos
        src2, _ = self.self_attn(q, k, value=src, need_weights=False)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src


class CachedTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )

    def forward(self, src, pos: Optional[torch.Tensor] = None):
        x = src
        outputs = []
        for layer in self.layers:
            x = layer(x, pos)
            outputs.append(x)
        return outputs


class TimestepConditionPool(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.token_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        timestep_emb: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query_norm(timestep_emb).unsqueeze(1)
        tokens = self.token_norm(tokens)
        pooled, _ = self.attn(query, tokens, tokens, need_weights=False)
        return pooled + timestep_emb.unsqueeze(1)


class LayerWiseAdaLNTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: nn.Module, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )

    def _expand_conditions(self, memory):
        conds = list(memory) if isinstance(memory, (list, tuple)) else [memory]
        if len(conds) == 0:
            raise RuntimeError("Decoder condition list is empty.")
        if len(conds) > len(self.layers):
            conds = conds[-len(self.layers):]
        if len(conds) < len(self.layers):
            conds = conds + [conds[-1]] * (len(self.layers) - len(conds))
        return conds

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ):
        x = tgt
        for layer, cond in zip(self.layers, self._expand_conditions(memory)):
            x = layer(
                x,
                cond,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                is_causal=tgt_is_causal,
            )
        return x


class FinalAdaLNLayer(nn.Module):
    def __init__(self, hidden_size: int, out_size: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(hidden_size, out_size)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)

class AdaLNDecoderLayer(nn.Module):
    """
    DiT-style decoder block.
    cond -> beta/scale/gate for self-attention and MLP.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_mlp = nn.Dropout(dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )

    def _condition_vector(self, cond: torch.Tensor) -> torch.Tensor:
        if cond is None:
            raise RuntimeError("Condition tokens are required for AdaLN decoder.")
        if cond.dim() != 3 or cond.shape[1] != 1:
            raise RuntimeError(
                f"Expected condition shape (B, 1, C), got {tuple(cond.shape)}."
            )
        return cond.squeeze(1)

    def _self_attention(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> torch.Tensor:
        try:
            out, _ = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
                is_causal=is_causal,
            )
        except TypeError:
            out, _ = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
        return out

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        cond = self._condition_vector(cond)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(cond).chunk(6, dim=-1)
        )

        attn_in = modulate(self.norm1(x), shift_attn, scale_attn)
        attn_out = self._self_attention(
            attn_in,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
        )
        x = x + gate_attn.unsqueeze(1) * self.dropout_attn(attn_out)

        mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.dropout_mlp(self.mlp(mlp_in))
        return x


class TransformerForDiffusion(ModuleAttrMixin):
    def __init__(self,
            input_dim: int,
            output_dim: int,
            horizon: int,
            n_obs_steps: int = None,
            cond_dim: int = 0,
            n_layer: int = 12,
            n_head: int = 12,
            n_emb: int = 768, # token dim (cond_dim -> n_emb 변환)
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            causal_attn: bool=False,
            time_as_cond: bool=True,
            obs_as_cond: bool=False,
            n_cond_layers: int = 0,
            n_cond_tokens: Optional[int] = None,
        ) -> None:
        super().__init__()

        # compute number of tokens for main trunk and condition encoder
        if n_obs_steps is None:
            n_obs_steps = horizon
        if not time_as_cond:
            raise NotImplementedError(
                "AdaLN-Zero decoder uses timestep embedding as its condition, "
                "so time_as_cond must be True."
            )
        

        T = horizon
        T_cond = 0
        if obs_as_cond:
            assert time_as_cond
            if n_cond_tokens is not None:
                T_cond = n_cond_tokens
            else:
                # fallback: one image token and one lowdim token per obs step
                T_cond = n_obs_steps * (1 + 1)

       
        # input embedding stem
        self.input_emb = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(input_dim, n_emb)
        ) # (T, action_dim) -> (T, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb)) # action position emb
        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        self.time_emb = SinusoidalPosEmb(n_emb) # denoising timestep
        self.time_mlp = nn.Sequential(
            nn.Linear(n_emb, n_emb),
            nn.SiLU(),
            nn.Linear(n_emb, n_emb)
        )
        self.cond_obs_emb = None
        
        # if obs_as_cond:
        #     self.cond_obs_emb = nn.Linear(cond_dim, n_emb) # (To, obs_dim) -> (To, n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.cond_pool = None
        self.decoder = None
        encoder_only = False
        if T_cond > 0: # True
            del self.cond_pos_emb
            self.register_buffer(
                "cond_pos_emb",
                build_sinusoidal_pos_emb(T_cond, n_emb),
                persistent=False
            )
            # encoder : condition 끼리 attention
            if n_cond_layers > 0: # False 
                encoder_layer = ConditionSelfAttnEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4*n_emb,
                    dropout=p_drop_attn,
                    activation='gelu'
                )
                self.encoder = CachedTransformerEncoder( # encoder가 Transformer + layer cache
                    encoder_layer=encoder_layer,
                    num_layers=n_cond_layers
                )
            else: # True
                self.encoder = nn.Sequential( # encoder가 MLP, 서로 안봄
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb)
                )
            self.cond_pool = TimestepConditionPool(
                d_model=n_emb,
                nhead=n_head,
                dropout=p_drop_attn,
            )

        # decoder : action self attention + adaLN-Zero conditioning
        decoder_layer = AdaLNDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4*n_emb,
            dropout=p_drop_attn,
            activation='gelu',
        )
        self.decoder = LayerWiseAdaLNTransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layer
        )

        # attention mask
        if causal_attn: # True
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask) # action self attention mask
        else:
            self.mask = None

        # decoder head
        self.final_layer = FinalAdaLNLayer(n_emb, output_dim)
            
        # constants
        self.T = T
        self.T_cond = T_cond
        self.n_emb = n_emb
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only

        # init
        self.apply(self._init_weights)
        self._init_adaln_zero()
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        ignore_types = (nn.Dropout, 
            SinusoidalPosEmb, 
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            CachedTransformerEncoder,
            LayerWiseAdaLNTransformerDecoder,
            FinalAdaLNLayer,
            ConditionSelfAttnEncoderLayer,
            TimestepConditionPool,
            AdaLNDecoderLayer,
            nn.ModuleList,
            nn.Identity,
            nn.SiLU,
            nn.GELU,
            nn.Mish,
            nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.xavier_uniform_(module.pos_emb)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def _init_adaln_zero(self):
        if self.decoder is None:
            return

        for layer in self.decoder.layers:
            if isinstance(layer, AdaLNDecoderLayer):
                mod_linear = layer.adaLN_modulation[-1]
                zero_linear_chunk(mod_linear, chunk_index=2, num_chunks=6)
                zero_linear_chunk(mod_linear, chunk_index=5, num_chunks=6)

    def _get_cond_pos_emb(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.cond_pos_emb is not None and length <= self.cond_pos_emb.shape[1]:
            return self.cond_pos_emb[:, :length, :].to(device=device, dtype=dtype)

        return build_sinusoidal_pos_emb(length, self.n_emb).to(
            device=device,
            dtype=dtype,
        )
    
    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if isinstance(self.cond_pos_emb, nn.Parameter):
            no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups


    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-6,
            betas: Tuple[float, float]=(0.95,0.999)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer


    def forward(self,   # model_output = model(trajectory, t, cond)로 호출
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        cond: Optional[torch.Tensor]=None, **kwargs): # cond: (B, token_num, token_feature)
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim)
        output: (B,T,input_dim)
        """
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        else:
            timesteps = timesteps.to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_mlp(self.time_emb(timesteps)) # (B,n_emb)
        

        # process input
        input_emb = self.input_emb(sample) # (B, T, n_emb)

        if not self.encoder_only: # True
            # encoder
            if self.obs_as_cond: # True
                if cond is None:
                    raise RuntimeError("Condition tokens must be provided when obs_as_cond=True.")
                cond_obs_emb = cond # (B, token_num, n_emb)
                cond_embeddings = cond_obs_emb

                if cond_embeddings.shape[-1] != self.n_emb:
                    raise RuntimeError(
                        f"Condition token dim must match n_emb={self.n_emb}, "
                        f"got {cond_embeddings.shape[-1]}."
                    )

                tc = cond_embeddings.shape[1] # token num
                position_embeddings = self._get_cond_pos_emb(
                    tc,
                    device=cond_embeddings.device,
                    dtype=cond_embeddings.dtype,
                )
                if isinstance(self.encoder, CachedTransformerEncoder):
                    x = self.drop(cond_embeddings)
                    x = self.encoder(x, position_embeddings)
                else:
                    x = self.drop(cond_embeddings + position_embeddings)
                    x = self.encoder(x)
                if isinstance(x, (list, tuple)):
                    encoder_outputs = list(x)
                else:
                    encoder_outputs = [x]
                decoder_conds = [
                    self.cond_pool(encoder_output, time_emb)
                    for encoder_output in encoder_outputs
                ]
            else:
                decoder_conds = [
                    time_emb.unsqueeze(1)
                    for _ in range(len(self.decoder.layers))
                ]
            
            # decoder
            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T,n_emb)

            # variable length
            tgt_mask = self.mask[:t, :t] if self.mask is not None else None

            x = self.decoder( # action self attention + precomputed adaLN cond
                tgt=x,
                memory=decoder_conds,
                tgt_mask=tgt_mask
            )
            # (B,T,n_emb)
        
        # head
        x = self.final_layer(x, decoder_conds[-1].squeeze(1))
        # (B,T,n_out)
        return x


def test():
    # GPT with time embedding
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        n_emb=16,
        n_head=4,
        # cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    out = transformer(sample, timestep)
    

    # GPT with time embedding and obs cond
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        n_emb=16,
        n_head=4,
        causal_attn=True,
        obs_as_cond=True,
        # time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()
    
    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    cond = torch.zeros((4,8,16))
    out = transformer(sample, timestep, cond)

    # GPT with time embedding and obs cond and encoder
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        n_emb=16,
        n_head=4,
        causal_attn=True,
        obs_as_cond=True,
        # time_as_cond=False,
        n_cond_layers=4
    )
    opt = transformer.configure_optimizers()
    
    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    cond = torch.zeros((4,8,16))
    out = transformer(sample, timestep, cond)

    # time-only adaLN conditioning
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        n_emb=16,
        n_head=4,
        # cond_dim=10,
        # causal_attn=True,
        time_as_cond=True,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    out = transformer(sample, timestep)
