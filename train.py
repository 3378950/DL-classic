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
        net = ResNet.resnet34(in_dim=3, num_classes=classes).to(device)

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

    def trainer(n_epochs):
        net.train()
        loss_over_time = []  # to track the loss as the network trains

        for epoch in range(n_epochs):  # loop over the dataset multiple times
            if saved_epoch:
                output_epoch = epoch + saved_epoch
            else:
                output_epoch = epoch

            running_loss = 0.0

            for batch_i, data in enumerate(train_loader):
                # get the input images and their corresponding labels
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter (weight) gradients
                optimizer.zero_grad()

                # forward pass to get outputs
                outputs = net(inputs)

                # calculate the loss
                loss = criterion(outputs, labels.long().to(device))
                # backward pass to calculate the parameter gradients
                loss.backward()

                # update the parameters
                optimizer.step()

                # print loss statistics
                # to convert loss into a scalar and add it to running_loss, we use .item()
                running_loss += loss.item()

                if batch_i % 45 == 44:  # print every 45 batches
                    avg_loss = running_loss / 45
                    # record and print the avg loss over the 100 batches
                    loss_over_time.append(avg_loss)
                    print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(output_epoch + 1, batch_i + 1, avg_loss))
                    running_loss = 0.0

            if output_epoch == 49 or output_epoch == 99:  # save every 100 epochs
                torch.save(net.state_dict(), './Models/saved_models/{}_{}.pt'.format(model, output_epoch + 1))

        print('Finished Training')
        return loss_over_time

    if saved_epoch:
        net.load_state_dict(torch.load('./saved_models/{}_{}.pt'.format(model, saved_epoch)))

    # call train and record the loss over time
    training_loss = trainer(n_epochs)

    
    # initialize tensor and lists to monitor test loss and accuracy
    test_loss = torch.zeros(1).to(device)
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    # set the module to evaluation mode
    # used to turn off layers that are only useful for training
    # like dropout and batch_norm
    net.eval()

    for batch_i, data in enumerate(test_loader):

        # get the input images and their corresponding labels
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # forward pass to get outputs
        outputs = net(inputs)

        # calculate the loss
        loss = criterion(outputs, labels.long().to(device))

        # update average test loss
        test_loss = test_loss + ((torch.ones(1).to(device) / (batch_i + 1)) * (loss.data - test_loss))

        # get the predicted class from the maximum value in the output-list of class scores
        _, predicted = torch.max(outputs.data, 1)

        # compare predictions to true label
        # this creates a `correct` Tensor that holds the number of correctly classified images in a batch
        correct = np.squeeze(predicted.eq(labels.data.view_as(predicted)))

        # calculate test accuracy for *each* object class
        # we get the scalar value of correct items for a class, by calling `correct[i].item()`
        for l, c in zip(labels.data, correct):
            class_correct[l] += c.item()
            class_total[l] += 1

    print('Test Loss: {:.6f}\n'.format(test_loss.cpu().numpy()[0]))

    for i in range(len(classes)):
        if class_total[i] > 0:
            print('Test Accuracy of %30s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))




