import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = datasets.CIFAR10(root='./data', train=True,
                                 download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False,
                                download=True, transform=transform)
validation_dataset, test_dataset = random_split(test_dataset, [0.5, 0.5])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# class CNN(nn.Module):
#     def __init__(self, num_classes=10):
#         super(CNN, self).__init__()

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)

#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.bn4 = nn.BatchNorm2d(128)

#         self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.bn5 = nn.BatchNorm2d(256)
#         self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.bn6 = nn.BatchNorm2d(256)

#         self.downsample1 = nn.Conv2d(64, 128, kernel_size=1, stride=2)
#         self.downsample2 = nn.Conv2d(128, 256, kernel_size=1, stride=2)
        
#         self.maxpool = nn.MaxPool2d(2, 2)
        
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
#         self.fc1 = nn.Linear(256, 512)
#         self.fc2 = nn.Linear(512, num_classes)
        
#         self.dropout = nn.Dropout(p=0.5)

#     def forward(self, x):
#         identity = x
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.relu(self.bn2(self.conv2(out)))
#         out = self.maxpool(out)
#         out += identity  

#         identity = self.downsample1(out)  
#         out = self.relu(self.bn3(self.conv3(out)))
#         out = self.relu(self.bn4(self.conv4(out)))
#         out = self.maxpool(out)
#         out += identity  

#         identity = self.downsample2(out) 
#         out = self.relu(self.bn5(self.conv5(out)))
#         out = self.relu(self.bn6(self.conv6(out)))
#         out = self.maxpool(out)
#         out += identity 

#         out = self.adaptive_pool(out)
#         out = torch.flatten(out, 1)
#         out = self.dropout(out)
#         out = self.fc1(out)
#         out = self.relu(out)
#         out = self.dropout(out)
#         out = self.fc2(out)

#         return out

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        self.relu = nn.ReLU(inplace=False)
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)

        self.downsample1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            self.maxpool,
        )

        self.downsample2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2),
            nn.BatchNorm2d(128)
        )
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.downsample3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2),
            nn.BatchNorm2d(256)
        )
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        identity = self.downsample1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.maxpool(out)
        out = out + identity

        identity = self.downsample2(out)
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.relu(self.bn4(self.conv4(out)))
        out = out + identity  

        identity = self.downsample3(out)
        out = self.relu(self.bn5(self.conv5(out)))
        out = self.relu(self.bn6(self.conv6(out)))
        out = out + identity 

        out = self.adaptive_pool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
best_model_dict = None
best_val_loss = float('inf')
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X, y in train_loader:
        optimizer.zero_grad()
        X = X.to(device)
        y = y.to(device)
        y_pred = model(X)
        loss = criterion(y_pred, y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader)
    
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for X, y in validation_loader:
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            val_loss += loss.item()
        val_loss /= len(validation_loader)

        if best_val_loss > val_loss:
            best_val_loss = val_loss
            best_model_dict = model.state_dict()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

torch.save(best_model_dict, 'best_model_dict.pt')
best_model = CNN().to(device)
best_model.load_state_dict(torch.load('best_model_dict.pt'))

best_model.eval()
with torch.no_grad():
    y_pred = []
    y_true = []
    for X, y in test_loader:
        X = X.to(device)
        y_pred.extend(best_model(X).argmax(dim=1).cpu().numpy())
        y_true.extend(y.cpu().numpy())
    
print(classification_report(y_true, y_pred))