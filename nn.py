"""
Neural network for Spades: policy + value heads.
Small ResNet, trainable on T4, inference on CPU via ONNX.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from features import FEATURE_DIM, BID_FEATURE_DIM, NUM_CARDS


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.relu(x + residual)
        return x


class PlayModel(nn.Module):
    """
    Input: state features (FEATURE_DIM)
    Output: policy over 52 cards + scalar value
    ~2.5M params
    """
    def __init__(self, input_dim: int = FEATURE_DIM, hidden: int = 256, num_blocks: int = 4):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(num_blocks)])

        # Policy head
        self.policy_fc1 = nn.Linear(hidden, 128)
        self.policy_fc2 = nn.Linear(128, NUM_CARDS)

        # Value head
        self.value_fc1 = nn.Linear(hidden, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.input_fc(x))
        for block in self.blocks:
            x = block(x)

        # Policy
        p = F.relu(self.policy_fc1(x))
        p = self.policy_fc2(p)  # raw logits, mask before softmax

        # Value
        v = F.relu(self.value_fc1(x))
        v = torch.tanh(self.value_fc2(v))

        return p, v

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class BidModel(nn.Module):
    """
    Input: hand + context features (BID_FEATURE_DIM)
    Output: distribution over bids 0-13
    """
    def __init__(self, input_dim: int = BID_FEATURE_DIM, hidden: int = 128, num_blocks: int = 2):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(num_blocks)])
        self.out_fc1 = nn.Linear(hidden, 64)
        self.out_fc2 = nn.Linear(64, 14)  # bids 0-13

    def forward(self, x):
        x = F.relu(self.input_fc(x))
        for block in self.blocks:
            x = block(x)
        x = F.relu(self.out_fc1(x))
        x = self.out_fc2(x)  # logits
        return x


def export_onnx(model: nn.Module, path: str, input_dim: int):
    """Export model to ONNX for fast CPU inference."""
    model.eval()
    dummy = torch.randn(1, input_dim)
    torch.onnx.export(
        model, dummy, path,
        input_names=['features'],
        output_names=['policy', 'value'] if isinstance(model, PlayModel) else ['bid_logits'],
        dynamic_axes={'features': {0: 'batch'}},
        opset_version=13
    )


def load_play_model(path: str, device='cpu') -> PlayModel:
    model = PlayModel()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def load_bid_model(path: str, device='cpu') -> BidModel:
    model = BidModel()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
