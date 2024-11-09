import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np


# 定义AutoEncoder模型
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # self.encoder = nn.Sequential(*list(resnet.children())[:-4])#去掉最后两层全连接层
        self.encoder = nn.Sequential(*list(resnet.children())[:-2]) 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  
            nn.Sigmoid()  # 将输出限制在[0, 1]之间
        )

    def forward(self, x):
        x = self.encoder(x)
        # return x

        x = self.decoder(x)
        x = nn.functional.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)  # 调整输出大小
        return x
