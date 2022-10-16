import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, in_dim, n_class) -> None:
        super().__init__()
        # Conv layer
        self.conv = nn.Sequential(
            # kernel: (in_dim x 11 x 11)
            nn.Conv2d(
                # the channel size
                in_channels=in_dim,
                out_channels=96,
                kernel_size=11, 
                stride=4,
                padding=0
            ),
            # feature map: (in_dim x 227 x 227) -> (96 x 55 x 55)

            nn.BatchNorm2d(96),
            nn.ReLU(True),
            # parms: nn.MaxPool2d(kernel_size, stride)
            nn.MaxPool2d(3, 2),
            # feature map: (96 x 55 x 55) -> (96 x 27 x 27)

            # kernel: (96 x 256 x 256)
            nn.Conv2d(96, 256, 5, stride=1, padding=2),
            # feature map: (96 x 27 x 27) -> (256 x 27 x 27)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            # feature map: (256 x 27 x 27) -> (256 x 13 x 13)

            # kernel: (384 x 3 x 3)
            nn.Conv2d(256, 384, 3, stride=1, padding=1),
            # feature map: (256 x 13 x 13) -> (384 x 13 x 13)
            nn.BatchNorm2d(384),
            nn.ReLU(True),

            # kernel: (384 x 3 x 3)
            nn.Conv2d(384, 384, 3, stride=1, padding=1),
            # feature map: (384 x 13 x 13) -> (384 x 13 x 13)
            nn.BatchNorm2d(384),
            nn.ReLU(True),

            # kernel: (256 x 3 x 3)
            nn.Conv2d(384, 256, 3, stride=1, padding=1),
            # feature map: (384 x 13 x 13) -> (256 x 13 x 13)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        
            nn.MaxPool2d(3, 2)
            # feature map: (256 x 13 x 13) -> (256 x 6 x 6)
        )
        
        # full-connective layer
        self.fc = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, n_class)
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # (batch,256,6,6) -> (batch,256*6*6)
        output = self.fc(x)
        return output