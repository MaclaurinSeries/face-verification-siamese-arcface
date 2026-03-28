import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetBackbone(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.features = base.features
        self.avgpool = base.avgpool
        self.embedder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self._init_embedder()

    def _init_embedder(self):
        nn.init.kaiming_normal_(self.embedder[1].weight)
        nn.init.zeros_(self.embedder[1].bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.embedder(x)
