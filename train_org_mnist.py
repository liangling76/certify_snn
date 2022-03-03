from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import time
from model_mnist import *
import sys

model = sys.argv[1]

if_train = True

if model == 'FC':
    checkpoint = './ckpt/model/mnist_FC.pth.tar'
    checkpoint_last = './ckpt/model/mnist_FC_last.pth.tar'
    snn = MNIST_FC().cuda()

elif model == 'CONV':
    checkpoint = './ckpt/model/mnist_CONV.pth.tar'
    checkpoint_last = './ckpt/model/mnist_CONV_last.pth.tar'
    snn = MNIST_CONV().cuda()


batch_size = 128

train_dataset = torchvision.datasets.MNIST(root='/home/lingliang/data', train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_set = torchvision.datasets.MNIST(root='/home/lingliang/data', train=False, download=True,  transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)


def lr_scheduler(optimizer, epoch, lr_decay_epoch=35):
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    return optimizer


optimizer = torch.optim.SGD(snn.parameters(), lr=1e-2, momentum=0.9)

criterion = nn.CrossEntropyLoss().cuda()  

num_epochs = 80
best_acc = 0  

if not if_train:
    num_epochs = 1
    snn.load_state_dict(torch.load(checkpoint))


for epoch in range(num_epochs):

    if if_train:
        snn.train()
        running_loss = 0
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            snn.zero_grad()
            optimizer.zero_grad()

            labels = labels.cuda()
            images = images.float().cuda()

            outputs = snn(images)
            loss = criterion(outputs, labels)
            running_loss += loss
            loss.backward()
            optimizer.step()

            if (i+1) % 300 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f' % (
                epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, running_loss / 300))
                running_loss = 0
                print('Time elasped:', time.time()-start_time)

    snn.eval()
    correct = 0
    total = 0
    optimizer = lr_scheduler(optimizer, epoch, 55)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images  = images.cuda()
            outputs = snn(images)

            loss = criterion(outputs, labels.cuda())

            _, predicted = outputs.cpu().max(1)
            total += float(labels.size(0))

            correct += float(predicted.eq(labels).sum().item())

    acc = 100. * float(correct) / float(total)
    print('Iters:', epoch)
    print('Test Accuracy of the model on the test images: %.3f' % acc)

    if if_train:
        if acc > best_acc:
            best_acc = acc
            print('Saving\n')
            torch.save(snn.state_dict(), checkpoint)

        if epoch == num_epochs - 1:
            torch.save(snn.state_dict(), checkpoint_last)