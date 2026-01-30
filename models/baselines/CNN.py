import torch
import torch.nn as nn
import torch.nn.functional as F

class Jet3DCNN_Binary(nn.Module):
    def __init__(self, num_layers=4):
        super(Jet3DCNN_Binary, self).__init__()

        # Input: (batch_size, 1, 15, 15, L)
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)

        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 1)  # 1 output for binary classification

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)           # (batch, 64, 1, 1, 1)
        x = x.view(x.size(0), -1)        # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                   # logits
        return x
