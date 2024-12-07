import torch
import torch.nn as nn


class SimpleTwoTower(nn.Module):
    def __init__(self, user_dim, episode_dim):
        super().__init__()

        self.user_layers = nn.Sequential(
            nn.Linear(user_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.episode_layers = nn.Sequential(
            nn.Linear(episode_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

    def forward(self, user_features, episode_features):
        user_features = user_features.to(self.user_layers[0].weight.dtype)
        episode_features = episode_features.to(self.user_layers[0].weight.dtype)
        user_emb = self.user_layers(user_features)
        episode_emb = self.episode_layers(episode_features)

        user_emb = user_emb / user_emb.norm(dim=1, keepdim=True)
        episode_emb = episode_emb / episode_emb.norm(dim=1, keepdim=True)
        scores = torch.sum(user_emb * episode_emb, dim=1)
        scores = scores.unsqueeze(dim=-1)

        return scores
