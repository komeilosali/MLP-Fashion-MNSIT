import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# MLP model:
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        #self.tanh = nn.Tanh()
        #self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        #out = self.tanh(out)
        #out = self.sigmoid(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Parameters
input_size = 28 * 28
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# Load the dataset
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transforms.ToTensor())

# Data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Create MLP model
model = MLP(input_size, hidden_size, num_classes)

# Define optimizer and loss function
criterion = nn.CrossEntropyLoss()

#optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=True)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=False)
#optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, input_size)

        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the training process 
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step,loss.item()))

# Testing the model
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Print accuracy
    print('Accuracy: {} %'.format(100 * correct / total))
