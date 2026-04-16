from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)
def print_attn_mask_grid(mask: torch.Tensor, name: str = "mask"):
    """
    mask: (T,S) 또는 (T,T), 값이 0.0(허용) / -inf(차단) 형태라고 가정
    """
    if mask is None:
        print(f"{name}: None")
        return

    m = mask.detach().cpu()
    allowed = torch.isfinite(m) & (m == 0)   # 허용
    blocked = ~allowed                        # 차단

    T, S = m.shape
    print(f"\n{torch.name} shape = ({T}, {S})")
    print("    " + " ".join([f"{j:2d}" for j in range(S)]))
    for i in range(T):
        row = []
        for j in range(S):
            row.append(" O" if allowed[i, j] else " X")
        print(f"{i:2d}: " + "".join(row))

#####
class CustomizedTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    nn.TransformerDecoderLayer 확장 버전.
    - cross-attn need_weights=True/False 토글 가능
    - 마지막 cross-attn weight를 레이어 내부에 저장
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture_cross_attention_weights = False
        self.last_cross_attn_weights = None

    def enable_cross_attention_weights(self, enabled: bool = True):
        self.capture_cross_attention_weights = enabled
        if not enabled:
            self.last_cross_attn_weights = None

    def _mha_block(self, x, mem, attn_mask, key_padding_mask, is_causal: bool = False):
        try:
            x, attn_weights = self.multihead_attn(
                x,
                mem,
                mem,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=self.capture_cross_attention_weights,
                is_causal=is_causal,
            )
        except TypeError:
            # torch 1.12 계열: is_causal 미지원
            x, attn_weights = self.multihead_attn(
                x,
                mem,
                mem,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=self.capture_cross_attention_weights,
            )
        self.last_cross_attn_weights = attn_weights if self.capture_cross_attention_weights else None
        return self.dropout2(x)
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
        ) -> None:
        super().__init__()

        # compute number of tokens for main trunk and condition encoder
        if n_obs_steps is None:
            n_obs_steps = horizon
        
        ##### 임시
        raw_wrench = True
        no_image = False
        one_image = True

        T = horizon
        if obs_as_cond:
            assert time_as_cond
            # T_cond = n_obs_steps * (rgb + low_dim) + wrench + time 
            T_cond = n_obs_steps * (2 + 1) + 1 + 1  # 8

            if no_image:
                T_cond = n_obs_steps * (1) + 1 + 1  # 4
                if raw_wrench:
                    T_cond = n_obs_steps * (1) + 1 # 3
            if one_image:
                T_cond = n_obs_steps * (1 + 1) + 1 + 1  # 6
                if raw_wrench:
                    T_cond = n_obs_steps * (1 + 1) + 1 # 5

        
        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb) # (T, action_dim) -> (T, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb)) # action position emb
        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        self.time_emb = SinusoidalPosEmb(n_emb) # denoising timestep
        self.cond_obs_emb = None
        
        # if obs_as_cond:
        #     self.cond_obs_emb = nn.Linear(cond_dim, n_emb) # (To, obs_dim) -> (To, n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False
        if T_cond > 0: # True
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
            # encoder : condition 끼리 attention
            if n_cond_layers > 0: # False 
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4*n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                )
                self.encoder = nn.TransformerEncoder( # encoder가 Transformer
                    encoder_layer=encoder_layer,
                    num_layers=n_cond_layers
                )
            else: # True
                self.encoder = nn.Sequential( # encoder가 MLP, 서로 안봄
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb)
                )
            # decoder : action self attention + cross attention to condition
            decoder_layer = CustomizedTransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True # important for stability
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=n_layer
            )
        # else: # False
        #     # encoder only BERT
        #     encoder_only = True

        #     encoder_layer = nn.TransformerEncoderLayer(
        #         d_model=n_emb,
        #         nhead=n_head,
        #         dim_feedforward=4*n_emb,
        #         dropout=p_drop_attn,
        #         activation='gelu',
        #         batch_first=True,
        #         norm_first=True
        #     )
        #     self.encoder = nn.TransformerEncoder(
        #         encoder_layer=encoder_layer,
        #         num_layers=n_layer
        #     )

        # attention mask
        if causal_attn: # True
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask) # action self attention mask
            
            if time_as_cond and obs_as_cond:
                
                S = T_cond # time token + obs tokens
                # t, s = torch.meshgrid(
                #     torch.arange(T),
                #     torch.arange(S),
                #     indexing='ij'
                # )
                # mask = t >= (s-1) # add one dimension since time is the first token in cond; time token이 맨앞
                # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                # self.register_buffer('memory_mask', mask)

                mem_allow = torch.ones((T, S), dtype=torch.bool)  # True=허용, False=차단
                x_list = [(0, 2), (0, 4), (0, 6), (0, 7)]

                if no_image:
                    x_list = [(0, 2), (0, 3)]
                if one_image:
                    x_list = [(0, 2), (0, 4), (0, 5)]
                    if raw_wrench:
                        x_list = [(0, 2), (0, 4)]

                for ti, si in x_list:
                    if 0 <= ti < T and 0 <= si < S:
                        mem_allow[ti, si] = False

                memory_mask = torch.zeros((T, S), dtype=torch.float32)
                memory_mask = memory_mask.masked_fill(~mem_allow, float('-inf'))
                self.register_buffer("memory_mask", memory_mask)
                
                # (time, imageR1, imageR2, imageT1, imageT2, lowdim1, lowdim2, force2)
                # ( o       o        x        o        x        o        x        x  ) <- action1

                # print_attn_mask_grid(memory_mask, name="memory_mask")
                #      0  1  2  3  4  5  6  7
                # 0:   O  O  X  O  X  O  X  X
                # 1:   O  O  O  O  O  O  O  O
                # 2:   O  O  O  O  O  O  O  O
                # 3:   O  O  O  O  O  O  O  O
                # 4:   O  O  O  O  O  O  O  O
                # 5:   O  O  O  O  O  O  O  O
                # 6:   O  O  O  O  O  O  O  O
                # 7:   O  O  O  O  O  O  O  O
                # 8:   O  O  O  O  O  O  O  O
                # 9:   O  O  O  O  O  O  O  O
                # 10:  O  O  O  O  O  O  O  O
                # 11:  O  O  O  O  O  O  O  O
                # 12:  O  O  O  O  O  O  O  O
                # 13:  O  O  O  O  O  O  O  O
                # 14:  O  O  O  O  O  O  O  O
                # 15:  O  O  O  O  O  O  O  O

            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)
            
        # constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only

        # init
        self.apply(self._init_weights)
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
            nn.ModuleList,
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
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))
    
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
        if self.cond_pos_emb is not None:
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
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

#####
    # attention weight 확인
    def set_attention_capture(self, enabled: bool = True) -> None:
        """Decoder cross-attention weight 저장 on/off."""
        if self.decoder is None:
            return
        for layer in self.decoder.layers:
            if isinstance(layer, CustomizedTransformerDecoderLayer):
                layer.enable_cross_attention_weights(enabled=enabled)

    def get_attention_weights(self):
        """레이어별 cross-attention weights 반환."""
        if self.decoder is None:
            return []

        return [
            {
                'layer': i,
                'cross_attn': layer.last_cross_attn_weights,
            }
            for i, layer in enumerate(self.decoder.layers)
            if isinstance(layer, CustomizedTransformerDecoderLayer)
        ]
#####

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
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1) # (B,1,n_emb)
        

        # process input
        input_emb = self.input_emb(sample) # (B, T, n_emb)

        if self.encoder_only: # False
            assert cond is None, "Conditioning is not supported in encoder-only mode"
            # # BERT
            # token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            # t = token_embeddings.shape[1]
            # position_embeddings = self.pos_emb[
            #     :, :t, :
            # ]  # each position maps to a (learnable) vector
            # x = self.drop(token_embeddings + position_embeddings)
            # # (B,T+1,n_emb)
            # x = self.encoder(src=x, mask=self.mask)
            # # (B,T+1,n_emb)
            # x = x[:,1:,:]
            # # (B,T,n_emb)
        else:
            # encoder
            cond_embeddings = time_emb # timestep emb (B, 1, n_emb)
            if self.obs_as_cond: # True
                # cond_obs_emb = self.cond_obs_emb(cond) # cond_dim -> n_emb (B,To,n_emb) 이미 맞춰 들어옴
                # cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1) # timestep + obs emb
                
                cond_obs_emb = cond # (B, token_num, n_emb)
                cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1) # timestep + obs emb
                # 이미지, lowdim, wrench 순서 잘 맞춰라잉!!!!

            tc = cond_embeddings.shape[1] # token num
            position_embeddings = self.cond_pos_emb[ # positional emb (token 수만큼)
                :, :tc, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(cond_embeddings + position_embeddings)
            x = self.encoder(x) # MLP or Transformer
            memory = x
            # (B,T_cond,n_emb)
            
            # decoder
            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T,n_emb)
            x = self.decoder( # action self attention + cross attention to condition
                tgt=x,
                memory=memory,
                tgt_mask=self.mask,
                memory_mask=None
            )
            # (B,T,n_emb)
        
        # head
        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,n_out)
        return x


def test():
    # GPT with time embedding
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
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
        cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()
    
    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    cond = torch.zeros((4,4,10))
    out = transformer(sample, timestep, cond)

    # GPT with time embedding and obs cond and encoder
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        n_cond_layers=4
    )
    opt = transformer.configure_optimizers()
    
    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    cond = torch.zeros((4,4,10))
    out = transformer(sample, timestep, cond)

    # BERT with time embedding token
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        # cond_dim=10,
        # causal_attn=True,
        time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    out = transformer(sample, timestep)

