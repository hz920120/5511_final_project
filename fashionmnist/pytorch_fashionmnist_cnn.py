#import lib
import datetime

import torch
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize = (3,3))

import torchvision
import torchvision.transforms as transforms

import yaml

yamlPath = "config.yaml"

config = []
with open(yamlPath,'rb') as f:
    config = yaml.safe_load(f)

data_path = config['data_path']
epochs = config['epochs']
batch_size = config['batch_size']

#setup training set
#transforming the PIL Image to tensors
trainset = torchvision.datasets.FashionMNIST(root = data_path, train = True, download = True, transform = transforms.ToTensor())

#loading the training data from trainset
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle = True)

#sneak peak into the train data

#iterating into the data
dataiter = iter(trainloader)
images, labels = dataiter.next()

print(images.shape) #shape of all 4 images
print(images[1].shape) #shape of one image
print(labels[1].item()) #label number

#taking the first image from batch of 4 images

img = images[1]
print(type(img))

#convert the tensor to numpy for displaying the image
npimg = img.numpy()
print(npimg.shape)

#for displaying the image, shape of the image should be height * width * channels
npimg = np.transpose(npimg, (1, 2, 0))
print(npimg.shape)

np.squeeze(npimg).shape

#plt.figure(figsize = (2,2))
#plt.imshow(np.squeeze(npimg))
#plt.show()

classes = ('T-Shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot')

def imshow(img):

    npimg = img.numpy() #convert the tensor to numpy for displaying the image
    #plt.imshow(np.transpose(npimg, (1, 2, 0))) #for displaying the image, shape of the image should be height * width * channels
    #plt.show()

imshow(torchvision.utils.make_grid(images))
#print(' '.join(classes[labels[j]] for j in range(4)))

import torch.nn as nn

class FirstCNN(nn.Module):
    def __init__(self):
        super(FirstCNN, self).__init__()
        #single layer convolution
        self.conv1 = nn.Conv2d(1, 16, 3, padding = (1,1), stride = (2, 2))

    def forward(self, x):
        #execute forward pass
        x = self.conv1(x)
        return(x)

#create a object of class
net = FirstCNN()

#print summary of network
print(net)

#input shape
images.shape

#output from the convolution
out = net(images)
out.shape

#total parameters in a network
for param in net.parameters():
    print(param.shape, "Parameters")

#plotting the output of convolution, taking the first channel
out1 = out[0, 0, :, :].detach().numpy()
print(out1.shape)

#display the output in the first layer

#plt.imshow(out[0, 0, :, :].detach().numpy())
#plt.figure(figsize = (10,10), dpi = 2000)
#plt.show()

#looking into the layer output, second layer output

#plt.imshow(out[0, 1, :, :].detach().numpy())
#plt.figure(figsize = (10,10))
#plt.show()

#creating custom convolution network using two blocks of 2D convolution
class FirstCNN_v2(nn.Module):
    def __init__(self):
        super(FirstCNN_v2, self).__init__()
        #create a cnn using nn.sequential
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, 3),   # (N, 1, 32, 32) -> (N, 8, 30, 30)
            nn.Conv2d(8, 16, 3)   # (N, 8, 30, 30) -> (N, 16, 28, 28)
        )

    def forward(self, x):
        x = self.model(x)
        return x

#create a class object and pass the input images to it.
net = FirstCNN_v2()
out = net(images)
out.shape

#visualize the convolution layer
#plt.imshow(out[0, 0, :, :].detach().numpy())

#in this class, we will use AvgPool for averge pooling - sub sampling
class FirstCNN_v3(nn.Module):
    def __init__(self):
        super(FirstCNN_v3, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 6, 5),          # (N, 1, 32, 32) -> (N, 6, 28, 28)
            nn.AvgPool2d(kernel_size = 2, stride=2),   # (N, 6, 28, 28) -> (N, 6, 14, 14)
            nn.Conv2d(6, 16, 5),         # (N, 6, 14, 14) -> (N, 16, 10, 10)
            nn.AvgPool2d(2, stride=2)    # (N, 16, 10, 10) -> (N, 16, 5, 5)
        )

    def forward(self, x):
        x = self.model(x)
        return x


#create a class object and pass the input images to it.
net = FirstCNN_v3()
out = net(images)
out.shape

#visualize the layer outputs
#plt.imshow(out[0, 0, :, :].detach().numpy())
#plt.show()

#plt.imshow(out[0, 1, :, :].detach().numpy())
#plt.show()

#class implementing the lenet network
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size = 5), #(N, 1, 28, 28) -> (N, 6, 24, 24)
            nn.Tanh(),
            nn.AvgPool2d(2, stride = 2), #(N, 6, 24, 24) -> (N, 6, 12, 12)

            nn.Conv2d(6, 16, kernel_size = 5), #(N, 6, 12, 12) -> (N, 6, 8, 8)
            nn.Tanh(),
            nn.AvgPool2d(2, stride = 2)) #(N, 6, 8, 8) -> (N, 16, 4, 4)

        self.fc_model = nn.Sequential(
            nn.Linear(256, 120), # (N, 256) -> (N, 120)
            nn.Tanh(),
            nn.Linear(120, 84), # (N, 120) -> (N, 84)
            nn.Tanh(),
            nn.Linear(84, 10))  # (N, 84)  -> (N, 10))

    def forward(self, x):
        #print(x.shape)
        x = self.cnn_model(x)
        #print(x.shape)
        #print(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.fc_model(x)
        #print(x.shape)
        return x

net = LeNet()
#running the lenet cnn
out = net(images)

#printing the class probabilities for 4 different images
print(out)

#taking only the maximum value
max_values, pred_class = torch.max(out.data, 1)
print(pred_class)

#increase the batch size
#download the data again and set the train, test loader with different batch size
trainset = torchvision.datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

def evaluation(dataloader):
    #function to calculate the accuracy
    total, correct = 0, 0
    for data in dataloader:
        #get the input and labels from data
        inputs, labels = data
        outputs = net(inputs)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        #calculate the accuracy
        correct += (pred == labels).sum().item()

    return(100 * correct/total)

#create an object of LeNet class,
net = LeNet()

import torch.optim as optim

#define the loss function
loss_fn = nn.CrossEntropyLoss()
#using the adam optimizer for backpropagation
opt = optim.Adam(net.parameters())

loss_arr = []
loss_epoch_arr = []


starttime = datetime.datetime.now()
for epoch in range(epochs):

    for i, data in enumerate(trainloader, 0):

        inputs, labels = data

        #forward pass
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)

        #backward and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_arr.append(loss.item())

    loss_epoch_arr.append(loss.item())

    print('Epoch: %d/%d, Test acc: %0.2f, Train acc: %0.2f' % (epoch+1, epochs, evaluation(testloader), evaluation(trainloader)))

endtime = datetime.datetime.now()
seconds = (endtime - starttime).seconds
m, s = divmod(seconds, 60)
h, m = divmod(m, 60)
print("total training time : %02d:%02d:%02d" % (h, m, s))


print('Test acc: %0.2f, Train acc: %0.2f' % (evaluation(testloader), evaluation(trainloader)))
plt.plot(loss_epoch_arr)
plt.show()

