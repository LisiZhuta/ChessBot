import torch.nn as nn
import torch.nn.functional as F
from .residual import ResidualBlock
from config import *

class ChessValueAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=input_dim, 
            num_heads=4,
            batch_first=True  # Critical fix for dimension ordering
        )
        self.ff = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        q = self.q_proj(x.unsqueeze(1))
        k = self.k_proj(x.unsqueeze(1))
        attn_out, _ = self.attn(q, k, x.unsqueeze(1))
        return self.ff(x + attn_out.squeeze(1))

class ChessSupervised(nn.Module):
    def __init__(self, rl_training=False):
        super().__init__()
        self.rl_training = rl_training
        
        # Enhanced convolutional backbone
        self.conv_layers = nn.Sequential(
            nn.Conv2d(13, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64, 32),
            ResidualBlock(64, 32),  
            nn.Conv2d(64, 128, 3, padding=1),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(4)
        )
        
        # Policy head with dropout
        self.policy_head = nn.Sequential(
            nn.Linear(128*4*4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(MOVE_VOCAB))
        )
        
        # Attention-based value head
        self.value_head = ChessValueAttention(128*4*4)

    def forward(self, x, temp=1.0):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        
        policy = self.policy_head(x) / temp
        if self.rl_training:
            policy += torch.randn_like(policy) * 0.05
            
        value = self.value_head(x).squeeze(-1)
        return policy, value