import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffusion_policy.model.force.force_encoder import CausalConvForceEncoder, GRUForceEncoder


def test_force_encoders(batch_size=8, input_dim=6, seq_len=32, feature_dim=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    causalconv_encoder = CausalConvForceEncoder(input_dim=input_dim, feature_dim=feature_dim).to(device).eval()
    gru_encoder = GRUForceEncoder(input_dim=input_dim, feature_dim=feature_dim).to(device).eval()

    x_causalconv = torch.randn(batch_size, input_dim, seq_len, device=device)
    x_gru = torch.randn(batch_size, input_dim, seq_len, device=device)

    with torch.no_grad():
        y_causalconv = causalconv_encoder(x_causalconv)
        y_gru = gru_encoder(x_gru)

    print(f"CausalConv input shape:  {tuple(x_causalconv.shape)}")
    print(f"CausalConv output shape: {tuple(y_causalconv.shape)}")
    print(f"GRU input shape:         {tuple(x_gru.shape)}")
    print(f"GRU output shape:        {tuple(y_gru.shape)}")

    assert y_causalconv.shape[0] == batch_size, "CausalConv batch size mismatch"
    assert y_causalconv.shape[1] == feature_dim, "CausalConv feature dim mismatch"
    assert y_gru.shape == (batch_size, feature_dim, 1), "GRU output shape mismatch"
    assert torch.isfinite(y_causalconv).all(), "CausalConv output contains NaN/Inf"
    assert torch.isfinite(y_gru).all(), "GRU output contains NaN/Inf"

    print("All tests passed.")


if __name__ == "__main__":
    test_force_encoders()
