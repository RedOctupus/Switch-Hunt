"""DQN 卷积网络。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    DQN网络 v8.23
    - 输入: 7通道 (墙壁 + 鬼位置 + 玩家位置 + A*路径 + 危险区 + CD状态 + 光源归一化次数)
    """
    def __init__(self, channels=7, height=21, width=21, action_size=4):
        super().__init__()
        # V7: 输入通道改为7
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.feature_size = 64 * height * width

        self.fc1 = nn.Linear(self.feature_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)
