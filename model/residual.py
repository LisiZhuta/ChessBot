import torch.nn as nn

# In residual.py
class ResidualBlock(nn.Module):
    def __init__(self, channels, hidden_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, channels, 1)
        )
    
    def forward(self, x):
        return x + self.conv(x)