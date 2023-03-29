# imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Lenet5 architecture
class lenet(nn.Module):
    def __init__(self,in_channels = 1,num_classes = 10):
        super(lenet, self).__init__()
        self.in_channels = in_channels
        self.num_classes =num_classes

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, 
                           kernel_size = 5, stride = 1, padding = 0)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, 
                            kernel_size = 5, stride = 1, padding = 0)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 120, 
                            kernel_size = 5, stride = 1, padding = 0)
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)
        self.tanh = nn.Tanh()
        self.avgpool = nn.AvgPool2d(kernel_size = 2, stride = 2)

    def forward(self,x):
        
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.avgpool(x)
        
        x = self.conv2(x)
        x = self.tanh(x)
        x = self.avgpool(x)
        
        x = self.conv3(x)
        x = self.tanh(x)
        
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        return x

# hyperparameters
# just for architecture testing
learning_rate = 0.001
batch_size = 64
num_epoch = 5


# Load data
train_datasets = datasets.MNIST(root='dataset/',train =True,transform=torchvision.transforms.Compose(
        [torchvision.transforms.Resize(32), torchvision.transforms.ToTensor()]
    ), download = True)
train_loader = DataLoader(dataset=train_datasets, batch_size =batch_size,shuffle=True)
test_datasets = datasets.MNIST(root='dataset/',train =False,transform=torchvision.transforms.Compose(
        [torchvision.transforms.Resize(32), torchvision.transforms.ToTensor()]
    ), download = True)
test_loader = DataLoader(dataset= test_datasets,batch_size = batch_size, shuffle=True)


# initialize n/w
model = lenet().to(device)

# Loss and optimizers
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)


# train n/w
for epoch in range(num_epoch):
    for batch_idx , (data, target) in enumerate(train_loader):

        data=data.to(device = device)
        target = target.to(device = device)

        # forward propagation
        scores = model(data)
        loss = criterion(scores,target)

        # zero previous gradient
        optimizer.zero_grad()

        # backward propagation
        loss.backward()

        # optimizer step 
        optimizer.step()


# check accuracy
def check_accuracy(loader,model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device = device)
            y = y.to(device= device)

            scores = model(x)
            _,predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

            print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)* 100:.2f}')
    model.train()

check_accuracy(train_loader,model)
check_accuracy(test_loader,model)


 







