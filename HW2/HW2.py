#%%[markdown]
# STAT 6289 - Statistical Deep Learning - HW2
#
# * Name: Baldur Hua 
# * [Github link](https://github.com/BaldurHua/STAT6289-Deep-Learning)

#%%[markdown]
# Question 1

# %%
import torch
import torchvision
import torchvision.transforms as transforms
#%%
# Load Data Set
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 32

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=batch_size, 
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', 
                                       train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, 
                                         batch_size=batch_size, 
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 
net = CNN()

#%%
# Accuracy Function
import torch.optim as optim

def ModAcc(model, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    acc = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # Validation
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                scores = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(scores.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print("Epoch {}/{}, Test Accuracy: {:.3f}".format(epoch+1, epochs, correct/total))
        acc.append(correct / total)
    return acc

#%%
# Simple Dense neural network with 0 hidden layer
class mod0(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(30, 10)
         
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model0 = mod0()

#%%
# Simple Dense neural network with 1 hidden layer

class mod1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072, 512)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
        
         
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x) 
        return x

model1 = mod1()


#%%
# Simple Dense neural network with 2 hidden layers

class mod2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072, 512)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 512)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 10)
                
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.fc2(self.drop(x)) 
        x = self.drop2(self.act2(x))
        x = self.fc3(x)
        return x

model2 = mod2()


#%%
# Simple Dense neural network with 3 hidden layers

class mod3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072, 512)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 512)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(512, 10)
                
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.fc2(self.drop(x)) 
        x = self.drop2(self.act2(x))
        x = self.act3(self.fc3(x))
        x = self.fc4(self.drop3(x))
        return x

model3 = mod3()
#%%
# Simple Dense neural network with 4 hidden layers

class mod4(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072, 512)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 512)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(512, 512)
        self.act4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(512, 10)
                
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.fc2(self.drop(x)) 
        x = self.drop2(self.act2(x))
        x = self.act3(self.fc3(x))
        x = self.fc4(self.drop3(x))
        x= self.drop4(self.act4(x))
        x = self.fc5(x)
        return x

model4 = mod4()
# %%
acc_cnn_relu = ModAcc(net, 10)
acc_0 = ModAcc(model0, 10)
acc_1 = ModAcc(model1, 10)
acc_2 = ModAcc(model2, 10)
acc_3 = ModAcc(model3, 10)
acc_4 = ModAcc(model4, 10)
#%%
plt.style.use('default')
plt.plot(acc_cnn_relu, label='CNN')
plt.plot(acc_0, label='Simple Dense - 0 Hidden')
plt.plot(acc_1, label='Simple Dense - 1 Hidden')
plt.plot(acc_2, label='Simple Dense - 2 Hidden')
plt.plot(acc_3, label='Simple Dense - 3 Hidden')
plt.plot(acc_4, label='Simple Dense - 4 Hidden')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.1, 0.7])
plt.legend(loc='upper right')

#%%
print("ready to continue Q2", "-"*50)

#%%[markdown]
# Question 2

#%%
class CNN_sig(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 32, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.sigmoid(self.conv1(x)))
        x = self.pool(F.sigmoid(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x
 
net_sig = CNN_sig()

# %%
acc_cnn_sig = ModAcc(net_sig, 10)
#%%
plt.style.use('default')
plt.plot(acc_cnn_relu, label='CNN - ReLU')
plt.plot(acc_cnn_sig, label='CNN - Sigmoid')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.25, 0.7])
plt.legend(loc='upper right')

#%%
print("ready to continue", "-"*50)
#%%[markdown]
# Question 3

#%%
# CNN with dropout=0.5

class CNN_drop(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.drop = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x
 
net_drop = CNN_drop()

#%%
# Augmentated Data

aug_transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset_aug = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=aug_transform)

trainloader_aug = torch.utils.data.DataLoader(trainset_aug, 
                                          batch_size=batch_size, shuffle=True, num_workers=0) # num_workers=2 doesn't work

testset_aug = torchvision.datasets.CIFAR10(root='./data', 
                                       train=False,
                                       download=True, transform=aug_transform)

testloader_aug = torch.utils.data.DataLoader(testset_aug, 
                                         batch_size=batch_size, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#%%
# Accuracy Function for train and test
#%%
# Accuracy Function
def ModAcc2(model, epochs, dataloader_train, dataloader_test):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs): 
        train_correct = 0
        train_total = 0
        train_acc = []
        for i, data in enumerate(dataloader_train, 0):
        # get the inputs; data is a list of [inputs, labels]
            inputs, train_labels = data
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()
            
            _, predicted_train = torch.max(outputs.data, 1)
            train_total += train_labels.size(0)
            train_correct += (predicted_train == train_labels).sum().item()
        train_acc.append(train_correct / train_total)
    
    # Validation

        test_correct = 0
        test_total = 0
        test_acc = []
        with torch.no_grad():
            for data in dataloader_test:
                images, test_labels = data
                # calculate outputs by running images through the network
                scores = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted_test = torch.max(scores.data, 1)
                test_total += test_labels.size(0)
                test_correct += (predicted_test == test_labels).sum().item()
            test_acc.append(test_correct / test_total)

            print("Epoch {}/{}, Train Accuracy: {:.3f}, Test Accuracy:{:.3f}".format(
                epoch+1, epochs, train_correct / train_total, test_correct / test_total))
    
    return [train_acc, test_acc]

#%%
No_drop_No_Aug = ModAcc2(net, 100, trainloader, testloader)
No_drop_Aug = ModAcc2(net, 100, trainloader_aug, testloader_aug)
Drop_No_Aug = ModAcc2(net_drop, 100, trainloader, testloader)
Drop_Aug = ModAcc2(net_drop, 100, trainloader_aug, testloader_aug)
#%%
Train_No_drop_No_Aug = No_drop_No_Aug[0]
Train_No_drop_Aug = No_drop_Aug[0]
Train_Drop_No_Aug = Drop_No_Aug[0]
Train_Drop_Aug = Drop_Aug[0]

Test_No_drop_No_Aug = No_drop_No_Aug[1]
Test_No_drop_Aug = No_drop_Aug[1]
Test_Drop_No_Aug = Drop_No_Aug[1]
Test_Drop_Aug = Drop_Aug[1]

#%%
plt.style.use('default')
plt.plot(Train_No_drop_No_Aug, label='No Dropout, No Aug')
plt.plot(Train_No_drop_Aug, label='No Drop, Aug')
plt.plot(Train_Drop_No_Aug, label='Drop, No Aug')
plt.plot(Train_Drop_Aug, label='Drop, Aug')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train Set')
plt.ylim([0, 1])
plt.legend(loc='upper left')

#%%
#%%
plt.style.use('default')
plt.plot(Test_No_drop_No_Aug, label='No Dropout, No Aug')
plt.plot(Test_No_drop_Aug, label='No Drop, Aug')
plt.plot(Test_Drop_No_Aug, label='Drop, No Aug')
plt.plot(Test_Drop_Aug, label='Drop, Aug')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Set')
plt.ylim([0, 1])
plt.legend(loc='upper left')

#%%