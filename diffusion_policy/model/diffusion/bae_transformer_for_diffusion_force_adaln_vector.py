import copy
import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb


logger = logging.getLogger(__name__)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_activation(activation: str) -> nn.Module:
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU(approximate="tanh")
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


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
            memory_is_causal: bool = False):
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
            activation: str = "gelu"):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation_name = activation

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
        # cond는 AdaLN에 쓰는 single token: (B, 1, n_emb)
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
            is_causal: bool) -> torch.Tensor:
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
            is_causal: bool = False) -> torch.Tensor:
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


class TransformerForDiffusionVectorCond(ModuleAttrMixin):
    """
    DiT-style action denoiser.
    policy에서 만든 vector condition을 timestep embedding과 더해서
    모든 AdaLN decoder block의 condition으로 사용한다.

    sample: (B, T, input_dim)
    cond: (B, cond_dim)
    output: (B, T, output_dim)
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            horizon: int,
            cond_dim: int,
            n_layer: int = 8,
            n_head: int = 4,
            n_emb: int = 512,
            p_drop_emb: float = 0.0,
            p_drop_attn: float = 0.1,
            causal_attn: bool = True,
            time_as_cond: bool = True,
        ) -> None:
        super().__init__()

        # vector AdaLN condition은 timestep embedding과 같이 쓰는 구조이다.
        if not time_as_cond:
            raise NotImplementedError(
                "TransformerForDiffusionVectorCond requires time_as_cond=True."
            )

        T = horizon

        # input embedding stem
        self.input_emb = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(input_dim, n_emb)
        ) # (B, T, action_dim) -> (B, T, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb)) # action position emb
        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        self.time_emb = SinusoidalPosEmb(n_emb) # denoising timestep
        self.time_mlp = nn.Sequential(
            nn.Linear(n_emb, n_emb),
            nn.SiLU(),
            nn.Linear(n_emb, n_emb)
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, n_emb),
            nn.SiLU(),
            nn.Linear(n_emb, n_emb)
        ) # (B, cond_dim) -> (B, n_emb)

        # decoder : action self attention + adaLN-Zero conditioning
        decoder_layer = AdaLNDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation='gelu',
        )
        self.decoder = LayerWiseAdaLNTransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layer
        )

        # attention mask
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            mask = (torch.triu(torch.ones(T, T)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask) # action self attention mask
        else:
            self.mask = None

        # decoder head
        self.final_layer = FinalAdaLNLayer(n_emb, output_dim)

        # constants
        self.T = T
        self.cond_dim = cond_dim
        self.n_emb = n_emb
        self.horizon = horizon
        self.time_as_cond = time_as_cond

        # init
        self.apply(self._init_weights)
        self._init_adaln_zero()
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            LayerWiseAdaLNTransformerDecoder,
            FinalAdaLNLayer,
            AdaLNDecoderLayer,
            nn.ModuleList,
            nn.SiLU,
            nn.GELU,
            nn.Sequential,
        )
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
        elif isinstance(module, TransformerForDiffusionVectorCond):
            torch.nn.init.xavier_uniform_(module.pos_emb)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def _init_adaln_zero(self):
        for layer in self.decoder.layers:
            if isinstance(layer, AdaLNDecoderLayer):
                mod_linear = layer.adaLN_modulation[-1]
                torch.nn.init.zeros_(mod_linear.weight)
                if mod_linear.bias is not None:
                    torch.nn.init.zeros_(mod_linear.bias)

        final_mod_linear = self.final_layer.adaLN_modulation[-1]
        torch.nn.init.zeros_(final_mod_linear.weight)
        if final_mod_linear.bias is not None:
            torch.nn.init.zeros_(final_mod_linear.bias)
        torch.nn.init.zeros_(self.final_layer.linear.weight)
        if self.final_layer.linear.bias is not None:
            torch.nn.init.zeros_(self.final_layer.linear.bias)

    def get_optim_groups(self, weight_decay: float = 1e-3):
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
                fpn = "%s.%s" % (mn, pn) if mn else pn
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

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, (
            "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        )
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters %s were not separated into either decay/no_decay set!" % (
                str(param_dict.keys() - union_params),)
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

    def configure_optimizers(
            self,
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-6,
            betas: Tuple[float, float] = (0.95, 0.999)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(
            self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            cond: Optional[torch.Tensor] = None,
            **kwargs):
        """
        sample: (B, T, input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B, cond_dim)
        output: (B, T, output_dim)
        """
        if cond is None:
            raise RuntimeError("Vector condition must be provided.")
        if cond.dim() != 2 or cond.shape[-1] != self.cond_dim:
            raise RuntimeError(
                f"Expected condition shape (B, {self.cond_dim}), got {tuple(cond.shape)}."
            )

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

        time_emb = self.time_mlp(self.time_emb(timesteps)) # (B, n_emb)

        # 2. vector condition
        cond_emb = self.cond_mlp(cond) # (B, n_emb)
        decoder_cond = (time_emb + cond_emb).unsqueeze(1) # (B, 1, n_emb)

        # 3. process input
        input_emb = self.input_emb(sample) # (B, T, n_emb)
        t = input_emb.shape[1]
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(input_emb + position_embeddings)
        tgt_mask = self.mask[:t, :t] if self.mask is not None else None

        # 4. decoder
        x = self.decoder(
            tgt=x,
            memory=decoder_cond,
            tgt_mask=tgt_mask
        )

        # head
        x = self.final_layer(x, decoder_cond.squeeze(1))
        return x
