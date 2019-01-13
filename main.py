from model import Net
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
# download and load CIFAR dataset

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
)

train_set = torchvision.datasets.CIFAR10(
    root = './data',
    train = True,
    transform = transform,
    download = True # False if already downloaded the dataset
)

test_set = torchvision.datasets.CIFAR10(
    root = './data',
    train = False,
    transform = transform,
    download = True # False if already downloaded the dataset
)

trainloader = torch.utils.data.DataLoader(
    train_set, 
    batch_size = 4,
    shuffle = True,
    num_workers = 2
)

testloader = torch.utils.data.DataLoader(
    test_set,
    batch_size = 4,
    shuffle = False,
    num_workers = 2
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
          'horse', 'ship', 'truck')

# train the model
net = Net() 
Loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = Loss(outputs, labels)# log loss
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 2000 == 1999: # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')

# test the model on test_dataset
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the model on test images: %d %%' % (100 * correct / total))















