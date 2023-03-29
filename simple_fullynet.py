# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# create fully connected Network
class NN(nn.Module):
    def __init__(self,input_size , num_classes):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50,num_classes)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x= self.fc2(x)
        return x
    
# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 784
num_classes =10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

#Load Data
train_datasets = datasets.MNIST(root='dataset/',train =True,transform=transforms.ToTensor(), download = True)
train_loader = DataLoader(dataset=train_datasets, batch_size =batch_size,shuffle=True)
test_datasets = datasets.MNIST(root='dataset/',train =False,transform=transforms.ToTensor(), download = True)
test_loader = DataLoader(dataset= test_datasets,batch_size = batch_size, shuffle=True)

# Initialize Network
model = NN(input_size=input_size , num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)

# Train Network

for epoch in range(num_epochs):
    print(f"Epoch: {epoch}")
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Get to correct shape, 28x28->784
        # -1 will flatten all outer dimensions into one
        data = data.reshape(data.shape[0], -1) 

        # forward propagation
        scores = model(data)
        loss = criterion(scores, targets)

        # zero previous gradients
        optimizer.zero_grad()
        
        # back-propagation
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

# check accuracy on training & test to see how good our model

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
            x = x.to(device =device)
            y = y.to(device= device)
            x = x.reshape(x.shape[0],-1)

            scores = model(x)
            _,predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)* 100:.2f}')
    model.train()

check_accuracy(train_loader,model)
check_accuracy(test_loader,model)
