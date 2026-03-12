import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffusion_policy.model.force.force_encoder import CausalConvForceEncoder
import torch


obs_force_encoder = CausalConvForceEncoder(feature_dim=8)


random_input = torch.randn(3, 6, 32)  # (B, 6, T)
output = obs_force_encoder(random_input)
print("Output shape:", output.shape)  # 기대하는 출력 shape: (B, feature_dim, 1) -> (3, 8, 1)

print('input:', random_input)
print('Output:', output)
