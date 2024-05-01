import torch
import torch.nn as nn
import gzip
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from trainer_torch import Trainer
import matplotlib.pyplot as plt

def read_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8, offset=16)
        data = data.reshape(-1, 28, 28)  # 每张图像的大小为28x28像素
        return data

def read_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        buffer = f.read()
        labels = np.frombuffer(buffer, dtype=np.uint8, offset=8)
        return labels
def calculate_accuracy(predictions, labels):
    _, predicted = torch.max(predictions, 1)
    correct = torch.sum(predicted == labels).item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy


train_images = read_mnist_images('minst/train-images-idx3-ubyte.gz')
train_labels = read_mnist_labels('minst/train-labels-idx1-ubyte.gz')
test_images = read_mnist_images('minst/t10k-images-idx3-ubyte.gz')
test_labels = read_mnist_labels('minst/t10k-labels-idx1-ubyte.gz')
test_images_tensor = torch.tensor(test_images, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
train_images_tensor = torch.tensor(train_images, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)


test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)
train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)

batch_size = 128 
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=len(test_dataset), shuffle=True)
class net(nn.Module):
    def __init__(self,h = 128):
        super(net, self).__init__()
        self.l1 =nn.Linear(784,h)
        self.l2 = nn.Linear(h,10)
        self.sigmoid = nn.Sigmoid()
        self.layers = [self.l2, self.l1]

    def forward(self,input):
        x = input.reshape(-1,784)
        x = self.sigmoid(self.l1(x))
        x = self.l2(x)
        return x
    
def step(lr = 0.01,weight_decay=0,hidden = 128,load_dir = '',plot = False):
    model = net(hidden)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr = lr,weight_decay= weight_decay)
    trainer = Trainer(model=model,optimizer=optimizer,metric=calculate_accuracy,loss_fn = loss_fn)
    trainer.train(train_loader=train_loader,dev_loader=test_loader)
    if plot:
        trainer.plot()
    return trainer.train_loss

step(plot=True)
if __name__ == '__main__':
    lr_list = [1,0.5,0.25,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]
    weight_list = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]
    h_list = [(784-(i+1)*(784-10)/11)//1 for i in range(10)]
    iter = 10
    x = range(iter)

    for lr in lr_list:
        print('lr ={}'.format(lr))
        loss = step(lr=lr)
        plt.plot(x, loss, label='lr =={}'.format(lr))
    plt.legend()
    plt.title('loss with different lr')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    for w in weight_list:
        print('weight_decay ={}'.format(w))
        loss = step(weight_decay=w)
        plt.plot(x, loss, label='weight_decay =={}'.format(w))
    plt.legend()
    plt.title('loss with different weight_decay')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    for h in h_list:
        h = int(h)
        print('hidden ={}'.format(h))
        loss = step(hidden= h)
        plt.plot(x, loss, label='hidden =={}'.format(h))
    plt.legend()
    plt.title('loss with different hidden size')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()