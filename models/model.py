import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== Simple CNN for log-Mel =====

class MelCNN(nn.Module):
    def __init__(self, n_mels: int = 64, num_classes: int = 4, channels=(32, 64, 128), dropout=0.2):
        super().__init__()
        C1, C2, C3 = channels
        self.conv1 = nn.Conv2d(1, C1, kernel_size=(5,5), padding=2)
        self.bn1 = nn.BatchNorm2d(C1)
        self.conv2 = nn.Conv2d(C1, C2, kernel_size=(3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(C2)
        self.conv3 = nn.Conv2d(C2, C3, kernel_size=(3,3), padding=1)
        self.bn3 = nn.BatchNorm2d(C3)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(C3, num_classes)

    def forward(self, x):  # x: [B, n_mels, T]
        x = x.unsqueeze(1)  # [B, 1, n_mels, T]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, (2,2))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, (2,2))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1).squeeze(-1)  # [B, C]
        x = self.drop(x)
        return self.fc(x)

# ===== MLP for HuBERT pooled embeddings =====

class HubertMLP(nn.Module):
    def __init__(self, in_dim=768, hidden=(512, 128), num_classes=4, dropout=0.2):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, num_classes)
        )

    def forward(self, x):  # x: [B, 768]
        return self.net(x)


def build_model(kind: str, cfg: dict):
    if kind == "mel":
        n_mels = cfg["mel"]["n_mels"]
        return MelCNN(n_mels=n_mels,
                      num_classes=cfg["mel_model"]["num_classes"],
                      channels=tuple(cfg["mel_model"]["channels"]),
                      dropout=cfg["mel_model"]["dropout"])
    else:
        return HubertMLP(in_dim=768,
                         hidden=tuple(cfg["hubert_model"]["hidden_dims"]),
                         num_classes=cfg["hubert_model"]["num_classes"],
                         dropout=cfg["hubert_model"]["dropout"])