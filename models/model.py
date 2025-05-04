
#模型定义文件

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MiniXception(nn.Module):
    def __init__(self, num_classes=7, input_channels=1):
        super(MiniXception, self).__init__()

        # 添加图像预处理层
        self.preprocess = nn.Sequential(
            nn.AdaptiveAvgPool2d((48, 48)),  # 统一调整图像大小为48x48
            nn.Conv2d(input_channels, 1, kernel_size=1)  # 如果是RGB图像，将其转换为灰度图
        )

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.25)

        self.residual1 = self._make_residual_block(64, 128)
        self.residual2 = self._make_residual_block(128, 256)
        self.residual3 = self._make_residual_block(256, 512)

        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.dropout3 = nn.Dropout(0.5)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_residual_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            ChannelAttention(out_channels),  # 新增注意力
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            ChannelAttention(out_channels),  # 新增注意力
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Dropout(0.25)
        )

    def forward(self, x):
        # 预处理输入数据
        x = self.preprocess(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x