from __future__ import annotations

import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs):
        return self.block(inputs)


class TrafficSignNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.05),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs):
        features = self.features(inputs)
        return self.classifier(features)

