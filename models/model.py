import torch
import torch.nn as nn

class MelCNN(nn.Module):
    def __init__(self, n_mels: int, channels: list, dropout: float, num_classes: int):
        super(MelCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, channels[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(channels[2])
        self.pool = nn.MaxPool2d(2, 2)
        # Approximate output size after convolutions and pooling
        self.fc_input_size = channels[2] * (n_mels // 8) * (151 // 8)  # Adjust based on input shape
        self.fc = nn.Linear(self.fc_input_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, n_mels, T] -> [B, 1, n_mels, T]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = self.fc(x)
        return x

class HubertMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float, num_classes: int):
        super(HubertMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def build_model(cfg: dict):
    feature_type = cfg["feature_type"]
    if feature_type == "mel":
        model_cfg = cfg["mel_model"]
        return MelCNN(
            n_mels=cfg["mel"]["n_mels"],
            channels=model_cfg["channels"],
            dropout=model_cfg["dropout"],
            num_classes=model_cfg["num_classes"]
        )
    elif feature_type == "hubert":
        model_cfg = cfg["hubert_model"]
        return HubertMLP(
            input_dim=768,  # HuBERT feature dimension
            hidden_dims=model_cfg["hidden_dims"],
            dropout=model_cfg["dropout"],
            num_classes=model_cfg["num_classes"]
        )
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")