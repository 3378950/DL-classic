import torch.nn as nn
import torch


import torch.backends.cudnn as cudnn

import torch.optim as optim


from utils import load_datasets


def train_model(model, in_dim, classes, optim_type, loss_fn, batch_size, n_epochs, learning_rate, saved_epoch, device_name):

    device = torch.device(device_name)


    # Load network and use GPU
    net = None

    if model == "VGG":
        from models import VGG
        net = VGG.VGG16(in_dim, classes).to(device)

    elif model == "AlexNet":
        from models import AlexNet
        net = AlexNet.AlexNet(in_dim, classes).to(device)

    elif model == "ResNet":
        from models import ResNet
        net = ResNet.resnet34(in_dim=in_dim, num_classes=classes).to(device)

    print("dimension of input: {}, classes: {}.".format(in_dim, classes))

    print("using model: {}".format(model), "run on device: {}".format(device))

    print("epochs in total: {}, the optimizer: {}, batch size: {}, learning rate: {}".format(n_epochs, optim_type, batch_size, learning_rate))

    cudnn.benchmark = True

    # Load dataset
    train_data, test_data, train_loader, test_loader = load_datasets(input_size=227, batch_size=batch_size)

    
    criterion = None

    if loss_fn == "cross entropy":
        # cross entropy loss: you don't need to add softmax func
        criterion = nn.CrossEntropyLoss()
    elif loss_fn == "nll loss":
        # cross entropy loss combines softmax and nn.NLLLoss() in one single class. You need to add softmax func in your model
        criterion = nn.NLLLoss()


    # stochastic gradient descent with a small learning rate
    optimizer = None

    if optim_type == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    elif optim_type == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)


    def trainer(num_epochs):
        for epoch in range(num_epochs):
            if saved_epoch:
                output_epoch = epoch + saved_epoch
            else:
                output_epoch = epoch
            print('current epoch = {}'.format(output_epoch))
            for i, (images, labels) in enumerate(train_loader):
                train_accuracy_total = 0
                train_correct = 0
                train_loss_sum = 0
                net.train()
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)   # 计算模型的损失
                optimizer.zero_grad()            # 在做反向传播前先清除网络状态
                loss.backward()                  # 损失值进行反向传播
                optimizer.step()                 # 参数迭代更新

                train_loss_sum += loss.item()    # item()返回的是tensor中的值，且只能返回单个值（标量），不能返回向量，使用返回loss等
                _,predicts = torch.max(outputs.data,dim=1)  # 输出10类中最大的那个值
                train_accuracy_total += labels.size(0)
                train_correct += (predicts == labels).cpu().sum().item()
            test_acc = evaluate_accuracy(test_loader,model)
            print('epoch:{0},   loss:{1:.4f},   train accuracy:{2:.3f},  test accuracy:{3:.3f}'.format(
                    output_epoch, train_loss_sum/batch_size, train_correct/train_accuracy_total, test_acc))
        print('------------finish training-------------')
    
    def evaluate_accuracy(data_iter, model):
        '''
            模型预测精度
        '''
        total = 0
        correct = 0 
        with torch.no_grad():
            model.eval()
            for images, labels in data_iter:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _,predicts = torch.max(outputs.data, dim=1)
                total += labels.size(0)
                correct += (predicts == labels).cpu().sum().numpy()
        return  correct / total

    trainer(n_epochs)