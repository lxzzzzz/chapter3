import torch
import torch.nn as nn


class RoadsideCandidateFusion(nn.Module):
    """Fuse sparse LiDAR tokens with sampled 2D candidate scores."""

    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(channels + 1, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, lidar_features, candidate_scores=None):
        if candidate_scores is None:
            candidate_scores = lidar_features.new_zeros((lidar_features.shape[0], 1))
        if candidate_scores.ndim == 1:
            candidate_scores = candidate_scores.unsqueeze(-1)
        delta = self.proj(torch.cat([lidar_features, candidate_scores.to(lidar_features.dtype)], dim=-1))
        return lidar_features + delta
