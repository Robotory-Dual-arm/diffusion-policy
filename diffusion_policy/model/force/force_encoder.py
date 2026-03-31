import torch.nn as nn
from torch.nn import functional as F
from diffusion_policy.model.force.layers import CausalConv1D

def init_weights(modules):
    """
    Weight initialization from original SensorFusion Code
    """
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class CausalConvForceEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim, initailize_weights=True):
        """
        Force encoder taken from selfsupervised code
        """
        super().__init__()
        self.feature_dim = feature_dim

        self.frc_encoder = nn.Sequential(   # (B, wrench_component, wrench_hist) -> (B, feature_dim, 1)
            CausalConv1D(input_dim, 16, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            CausalConv1D(16, 32, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            CausalConv1D(32, 64, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            CausalConv1D(64, 128, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            CausalConv1D(128, self.feature_dim, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
        )

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, force):
        return self.frc_encoder(force).permute(0, 2, 1) # (B, 1, feature_dim)
    
class GRUForceEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim, initailize_weights=True):
        super().__init__()
        self.feature_dim = feature_dim

        self.gru = nn.GRU(input_size=input_dim, hidden_size=feature_dim, batch_first=True)

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, force):
        y, h = self.gru(force.permute(0, 2, 1))
        return h.permute(1, 0, 2) # (B, 1, feature_dim)
    
