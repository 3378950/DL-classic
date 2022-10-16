import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms as T


def vgg_block(in_channels, out_channels, num_convs, kernel_size=3, stride=1, padding=1):
    """
        in_channels: 输入通道数
        out_channels: 输出通道数
        num_convs: 卷积个数
        kernel_size: 卷积核大小
        stride: 步长
        padding: 填充大小
    """
    block = nn.Sequential()
    for i in range(num_convs):
        conv2d = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )
        block.add_module(f'conv2d_{i}',conv2d)
        in_channels = out_channels
    block.add_module(f'maxpool_{i}',nn.MaxPool2d(2, 2))
    return block

class VGG16(nn.Module):
    def __init__(self, in_dim, n_class):
        super().__init__()
        self.features = nn.Sequential(
            vgg_block(in_channels=in_dim, out_channels=64, num_convs=2),

            vgg_block(in_channels=64, out_channels=128, num_convs=2),
            
            vgg_block(in_channels=128, out_channels=256, num_convs=3),
            
            vgg_block(in_channels=256, out_channels=512, num_convs=3),

            vgg_block(in_channels=512, out_channels=512, num_convs=3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            
            nn.Linear(4096, n_class)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x