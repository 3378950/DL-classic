import torch 
import torch.nn as nn

def Basic_Conv2d(in_channels,out_channels,kernel_size,stride=1,**kwargs):
    '''
    基础卷积块:要求最后输出的各个路径feature map 大小相同
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                  stride=1, padding=kernel_size//2), 
        nn.BatchNorm2d(out_channels), 
        nn.ReLU(True)
    )

class Inception_V1(nn.Module):
    '''
        inception块搭建
    '''
    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, 
                 out_channels3reduce, out_channels3, out_channels4) -> None:
        super().__init__()
        
        # 线路1：单个1X1卷积
        self.branch1_conv = Basic_Conv2d(in_channels,out_channels1,kernel_size=1)
        
        # 线路2：1X1卷积层后接3X3卷积层
        self.branch2_conv1 = Basic_Conv2d(in_channels,out_channels2reduce,kernel_size=1)
        self.branch2_conv2 = Basic_Conv2d(out_channels2reduce,out_channels2,kernel_size=3)
        
        # 线路3：1X1卷积层后接5X5卷积层
        self.branch3_conv1 = Basic_Conv2d(in_channels,out_channels3reduce,kernel_size=1)
        self.branch3_conv2 = Basic_Conv2d(out_channels3reduce,out_channels3,kernel_size=5) 
        
        # 线路4：3X3最大池化后接1X1卷积层
        self.branch4_pool = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.branch4_conv1 = Basic_Conv2d(in_channels,out_channels4,kernel_size=1)

    def forward(self, x):
        out1 = self.branch1_conv(x)
        out2 = self.branch2_conv2(self.branch2_conv1(x))
        out3 = self.branch3_conv2(self.branch3_conv1(x))
        out4 = self.branch4_conv1(self.branch4_pool(x))
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

class GoogLenet_V1(nn.Module):
    def __init__(self, in_dim, n_class) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # 第一个模块
            nn.Conv2d(in_dim, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # 第二个模块
            nn.Conv2d(64,64,kernel_size=1),
            nn.Conv2d(64,192,kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            nn.MaxPool2d(3,2,1),
            
            # 第三个模块
            Inception_V1(in_channels=192, out_channels1=64, out_channels2reduce=96, out_channels2=128,
                         out_channels3reduce=16, out_channels3=32, out_channels4=32),
            Inception_V1(in_channels=256, out_channels1=128, out_channels2reduce=128, out_channels2=192,
                         out_channels3reduce=32, out_channels3=96, out_channels4=64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # 第四个模块
            Inception_V1(in_channels=480, out_channels1=192, out_channels2reduce=96, out_channels2=208,
                         out_channels3reduce=16, out_channels3=48, out_channels4=64),
            Inception_V1(in_channels=512, out_channels1=160, out_channels2reduce=112, out_channels2=224,
                         out_channels3reduce=24, out_channels3=64, out_channels4=64),
            Inception_V1(in_channels=512, out_channels1=128, out_channels2reduce=128, out_channels2=256,
                         out_channels3reduce=24, out_channels3=64, out_channels4=64),
            Inception_V1(in_channels=512, out_channels1=112, out_channels2reduce=144, out_channels2=288,
                         out_channels3reduce=32, out_channels3=64, out_channels4=64),
            Inception_V1(in_channels=528, out_channels1=256, out_channels2reduce=160, out_channels2=320,
                         out_channels3reduce=32, out_channels3=128, out_channels4=128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # 第五个模块
            Inception_V1(in_channels=832, out_channels1=256, out_channels2reduce=160, out_channels2=320,
                         out_channels3reduce=32, out_channels3=128, out_channels4=128),
            Inception_V1(in_channels=832, out_channels1=384, out_channels2reduce=192, out_channels2=384,
                         out_channels3reduce=48, out_channels3=128, out_channels4=128),
            nn.AvgPool2d(kernel_size=7, stride=1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024, n_class)
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x