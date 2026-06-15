"""
models/resnet.py — AlphaZero-style dual-headed ResNet in PyTorch.

Architecture:
  Input (batch, INPUT_CHANNELS, 8, 8)
  → Conv block
  → N residual blocks
  → Policy head → logits over 4672 moves
  → Value head  → tanh scalar in [-1, +1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_RES_BLOCKS, NUM_FILTERS, INPUT_CHANNELS, POLICY_SIZE


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                              padding=kernel//2, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        skip = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + skip)


class TaciturnNet(nn.Module):
    """
    Dual-headed ResNet.
    Input:  (batch, INPUT_CHANNELS, 8, 8)
    Output: policy_logits (batch, POLICY_SIZE), value (batch, 1)
    """

    def __init__(self):
        super().__init__()

        # Stem
        self.stem = ConvBnRelu(INPUT_CHANNELS, NUM_FILTERS)

        # Residual tower
        self.tower = nn.Sequential(
            *[ResBlock(NUM_FILTERS) for _ in range(NUM_RES_BLOCKS)]
        )

        # Policy head
        self.policy_conv = ConvBnRelu(NUM_FILTERS, 32, kernel=1)
        self.policy_fc   = nn.Linear(32 * 8 * 8, POLICY_SIZE)

        # Value head
        self.value_conv = ConvBnRelu(NUM_FILTERS, 1, kernel=1)
        self.value_fc1  = nn.Linear(8 * 8, 256)
        self.value_fc2  = nn.Linear(256, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.tower(x)

        # Policy
        p = self.policy_conv(x)
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value
        v = self.value_conv(x)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value


def build_model(device: torch.device) -> TaciturnNet:
    model = TaciturnNet().to(device)
    return model


def predict_single(model: TaciturnNet, board_tensor, device: torch.device):
    """
    Run inference on a single board tensor.
    Args:
        board_tensor: numpy (INPUT_CHANNELS, 8, 8)
    Returns:
        policy_probs: numpy (POLICY_SIZE,)
        value:        float
    """
    import numpy as np
    model.eval()
    with torch.no_grad():
        inp = torch.from_numpy(board_tensor[np.newaxis]).float().to(device)
        logits, val = model(inp)
        probs = torch.softmax(logits[0], dim=0).cpu().numpy()
        return probs, float(val[0, 0].cpu())
