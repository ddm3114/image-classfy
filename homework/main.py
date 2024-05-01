import activation,base,conv,loss,pooling,linear
import trainer
import sgd
import accuracy
import gzip
import numpy as np

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

train_images = read_mnist_images('minst/train-images-idx3-ubyte.gz')
train_labels = read_mnist_labels('minst/train-labels-idx1-ubyte.gz')
test_images = read_mnist_images('minst/t10k-images-idx3-ubyte.gz')
test_labels = read_mnist_labels('minst/t10k-labels-idx1-ubyte.gz')
dataset_test = (test_images,test_labels)
dataset_train = (train_images,train_labels)


class net(base.Module):
    def __init__(self):
        self.l1 =linear.Linear(784,128)
        self.l2 = linear.Linear(128,10)
        self.sigmoid1 = activation.Sigmoid()
        self.layers = [self.l2, self.l1]

    def forward(self,input):
        x = input.reshape(-1,784)
    
        print(x)
        x = self.sigmoid1(self.l1(x))
        x = self.l2(x)
        return x
    
   
    def backward(self,output_grad):
        grad = output_grad

        grad = self.l2.backward(grad)
        grad = self.sigmoid1.backward(grad)

        grad = self.l1.backward(grad)

model = net()
criterion = loss.CrossEntropyLoss(model=model,reduction='mean')
optimizer = sgd.SGD(model=model,lr=0.01)
trainers = trainer.Trainer(model=model,optimizer=optimizer,metric=accuracy.calculate_accuracy,loss_fn=criterion)
trainers.train(train_set=dataset_train,dev_set=dataset_test)