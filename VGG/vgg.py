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




def main():
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # batch大小
    batch_size = 64

    # 训练集和测试集的数据增强
    train_transform = T.Compose([
                        T.Resize((224,224)),
                        T.RandomHorizontalFlip(0.5),
                        T.ToTensor()
    ])
    test_transform = T.Compose([
                        T.Resize((224,224)),
                        T.ToTensor()
    ])

    # mnist数据集
    train_dataset = datasets.MNIST(root=r'data\mnist', 
                                    train = True, transform = train_transform, download = True)
    test_dataset = datasets.MNIST(root=r'data\mnist', 
                                    train = False, transform =test_transform)
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

    model = VGG16(1, 10).to(device)

    torch.manual_seed(1)
    # 学习率
    learning_rate = 1e-3
    # 训练轮数
    num_epochs = 10
    # 优化算法Adam = RMSProp + Momentum (梯度、lr两方面优化下降更快更稳)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    # 交叉熵损失函数
    loss_fn = torch.nn.CrossEntropyLoss()



    def evaluate_accuracy(data_iter, model):
        total = 0
        correct = 0 
        with torch.no_grad():
            model.eval()
            for images,labels in data_iter:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _,predicts = torch.max(outputs.data, dim=1)
                total += labels.size(0)
                correct += (predicts == labels).cpu().sum()
        return 100 * correct / total


    def train():
        for epoch in range(num_epochs):
            print('current epoch = {}'.format(num_epochs))
            for i,(images, labels) in enumerate(train_loader):
                train_accuracy_total = 0
                train_correct = 0
                train_loss_sum = 0
                model.train()
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs,labels)   # 计算模型的损失
                optimizer.zero_grad()            # 在做反向传播前先清除网络状态
                loss.backward()                  # 损失值进行反向传播
                optimizer.step()                 # 参数迭代更新

                train_loss_sum += loss.item()    # item()返回的是tensor中的值，且只能返回单个值（标量），不能返回向量，使用返回loss等
                _,predicts = torch.max(outputs.data,dim=1)  # 输出10类中最大的那个值
                train_accuracy_total += labels.size(0)
                train_correct += (predicts == labels).cpu().sum().item()
            test_acc = evaluate_accuracy(test_loader, model, device)
            print('epoch:{0},   loss:{1:.4f},   train accuracy:{2:.3f},  test accuracy:{3:.3f}'.format(
                    epoch, train_loss_sum/batch_size, train_correct/train_accuracy_total, test_acc))
        print('------------finish training-------------')

    train()


    
    

      


if __name__ == '__main__':
    main()