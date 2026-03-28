import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()

        self.conv1 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        return self.relu(x + identity)


class SimpleClassifier(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            # 224 x 224
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 112 x 112
            ResidualBlock(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 56 x 56
            ResidualBlock(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            # adaptive pooling
            # 4 x 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.Tanh(),
            nn.Dropout(0.4),
            nn.Linear(256, embed_dim),
        )
        self._init_weights()

    def _init_weights(self):
        def init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0, 0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if getattr(m, "weight", None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init)

    def forward(self, x):
        return self.classifier(self.features(x))
