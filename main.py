from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_cifar100(batch_size=128, shuffle=True):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        ]
    )
    
    # Load CIFAR-100 training and test datasets
    trainset = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform
    )

    testset = datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform
    )
    
    print(f"Training set size: {len(trainset)}")
    print(f"Test set size: {len(testset)}")
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def load_cifar10(batch_size=128, shuffle=True):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]),
        ]
    )
    
    # Load CIFAR-10 training and test datasets
    trainset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    
    print(f"Training set size: {len(trainset)}")
    print(f"Test set size: {len(testset)}")
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


class CustomCNN(torch.nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        # input size 3 colors 32x32 pixel
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.1),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.1),
        )
        self.fc_1 = nn.Sequential(
            nn.Linear(24 * 8 * 8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.fc_final = nn.Sequential(
            nn.Linear(64, 10),
        )
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = x.view(-1, 24 * 8 * 8)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_final(x)
        return x


def train(train_loader, model, optimiser, loss_fn, epochs):
    for epoch in tqdm(range(epochs), desc="Training"):
        running_loss = 0.0
        for x, label in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            # reset grad
            optimiser.zero_grad()
            label_pred = model(x)
            loss = loss_fn(label_pred, label)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
        
        # calcualte average loss per batch for final loss
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
    
    
def evaluate(test_loader, model):
    with torch.no_grad():
        correct_labels = 0
        num_items = test_loader[0].shape
        for x, label in tqdm(test_loader):
            label_pred = model(x)
            correct_labels += label == label_pred
        accuracy = correct_labels / num_items
        print(f"accuracy {accuracy}")
        


def main():
    batch_size=64
    epochs=2
    learning_rate=0.001
    
    train_loader, test_loader = load_cifar10(batch_size)

    model = CustomCNN()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    train(train_loader, model, optimiser, loss_fn, epochs)
    
    

    


if __name__ == "__main__":
    main()
