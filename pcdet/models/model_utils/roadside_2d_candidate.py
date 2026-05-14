import torch
import torch.nn as nn


class Roadside2DCandidateBranch(nn.Module):
    """Lightweight image candidate scorer for sparse roadside fusion experiments."""

    def __init__(self, in_channels, hidden_channels=64, score_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, score_channels, kernel_size=1),
        )

    def forward(self, image_features):
        logits = self.net(image_features)
        return {
            'candidate_logits': logits,
            'candidate_scores': torch.sigmoid(logits),
        }
