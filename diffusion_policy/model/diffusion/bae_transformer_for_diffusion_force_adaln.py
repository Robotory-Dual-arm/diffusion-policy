import copy
import math
import inspect
from typing import Union, Optional, Tuple, Any, cast
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

class ShiftScaleMod(nn.Module):
    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(cond_dim, hidden_dim)
        self.shift = nn.Linear(cond_dim, hidden_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cond = self.act(cond)
        scale = self.scale(cond).unsqueeze(1)
        shift = self.shift(cond).unsqueeze(1)
        return x * scale + shift


class ZeroScaleMod(nn.Module):
    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(cond_dim, hidden_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cond = self.act(cond)
        scale = self.scale(cond).unsqueeze(1)
        return x * scale


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

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU(approximate="tanh")
        else:
            raise RuntimeError(
                f"activation should be relu/gelu, not {activation}."
            )

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


class LayerWiseAdaLNTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: nn.Module, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )

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
        if isinstance(memory, (list, tuple)):
            conds = list(memory)
        else:
            conds = [memory]

        if len(conds) == 0:
            raise RuntimeError("Decoder condition list is empty.")
        if len(conds) > len(self.layers):
            conds = conds[-len(self.layers):]
        if len(conds) < len(self.layers):
            conds = conds + [conds[-1]] * (len(self.layers) - len(conds))

        x = tgt
        for layer, layer_cond in zip(self.layers, conds):
            x = layer(
                tgt=x,
                memory=layer_cond,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
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
        x = x * scale.unsqueeze(1) + shift.unsqueeze(1)
        return self.linear(x)

#####
class CustomizedTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    nn.TransformerDecoderLayer 확장 버전.
    - cross-attn 제거
    - forward에서 미리 만든 cond vector로 self-attn / FFN residual을 adaLN-Zero modulation
    """
    def __init__(self, *args, n_cond_tokens: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture_cross_attention_weights = False
        self.last_cross_attn_weights = None
        self.n_cond_tokens = n_cond_tokens

        d_model = self.self_attn.embed_dim
        cond_dim = d_model
        self.attn_mod1 = ShiftScaleMod(cond_dim, d_model)
        self.attn_mod2 = ZeroScaleMod(cond_dim, d_model)
        self.mlp_mod1 = ShiftScaleMod(cond_dim, d_model)
        self.mlp_mod2 = ZeroScaleMod(cond_dim, d_model)
        self.multihead_attn = nn.Identity() # parameter 제거
        self.norm3 = nn.Identity() # parameter 제거

    def enable_cross_attention_weights(self, enabled: bool = True):
        self.capture_cross_attention_weights = enabled
        if not enabled:
            self.last_cross_attn_weights = None

    def _get_cond(self, memory):
        self.last_cross_attn_weights = None
        if memory is None:
            raise RuntimeError("Condition tokens are required for adaLN-Zero decoder.")

        if memory.dim() != 3:
            raise RuntimeError(
                f"Expected condition tensor with shape (B, T_cond, C), got {tuple(memory.shape)}"
            )
        if memory.shape[1] != self.n_cond_tokens:
            raise RuntimeError(
                f"Expected {self.n_cond_tokens} condition tokens, got {memory.shape[1]}"
            )

        return memory.squeeze(1)

    def _sa_block_compat(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> torch.Tensor:
        """
        PyTorch version compatibility wrapper.
        - torch>=2.x: _sa_block(x, attn_mask, key_padding_mask, is_causal=...)
        - torch<=1.13: _sa_block(x, attn_mask, key_padding_mask)
        """
        sa_block = cast(Any, self._sa_block)
        sa_sig = inspect.signature(sa_block)
        if "is_causal" in sa_sig.parameters:
            return sa_block(
                x,
                attn_mask,
                key_padding_mask,
                is_causal,
            )

        return sa_block(
            x,
            attn_mask,
            key_padding_mask,
        )

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
        cond = self._get_cond(memory)
        if self.norm_first:
            x2 = self.attn_mod1(self.norm1(x), cond) # norm -> scale, shift
            x2 = self._sa_block_compat( # self-attn
                x2,
                tgt_mask,
                tgt_key_padding_mask,
                is_causal=tgt_is_causal,
            )
            x = x + self.attn_mod2(x2, cond) # scale

            x2 = self.mlp_mod1(self.norm2(x), cond) # norm -> scale, shift
            x2 = self._ff_block(x2) # FFN
            x = x + self.mlp_mod2(x2, cond) # scale
        else:
            x = self.norm1(
                x + self.attn_mod2(
                    self._sa_block_compat(
                        self.attn_mod1(x, cond),
                        tgt_mask,
                        tgt_key_padding_mask,
                        is_causal=tgt_is_causal,
                    ),
                    cond,
                )
            )
            x = self.norm2(
                x + self.mlp_mod2(
                    self._ff_block(self.mlp_mod1(x, cond)),
                    cond,
                )
            )
        return x
#####
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

        # decoder : action self attention + adaLN-Zero conditioning
        decoder_layer = CustomizedTransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4*n_emb,
            dropout=p_drop_attn,
            activation='gelu',
            n_cond_tokens=1,
            batch_first=True,
            norm_first=True # important for stability
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
            nn.ModuleList,
            nn.Identity,
            nn.SiLU,
            nn.GELU,
            nn.Mish,
            nn.Sequential,
            ShiftScaleMod,
            ZeroScaleMod)
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
            if isinstance(layer, CustomizedTransformerDecoderLayer):
                torch.nn.init.zeros_(layer.attn_mod2.scale.weight)
                torch.nn.init.zeros_(layer.attn_mod2.scale.bias)
                torch.nn.init.zeros_(layer.mlp_mod2.scale.weight)
                torch.nn.init.zeros_(layer.mlp_mod2.scale.bias)

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
                    (encoder_output.mean(dim=1) + time_emb).unsqueeze(1)
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
