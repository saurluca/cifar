from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torcheval.metrics import MulticlassAccuracy
import matplotlib.pyplot as plt


def load_cifar100(val_ratio, batch_size=128, shuffle=True, num_workers=4):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
            ),
        ]
    )

    # Load CIFAR-100 training and test datasets
    trainset = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform
    )

    testset = datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform
    )

    # Calculate split sizes
    val_size = int(len(trainset) * val_ratio)
    train_size = len(trainset) - val_size

    # Split training set into train and validation
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

    print(f"Training set size: {len(trainset)}")
    print(f"Validation set size: {len(valset)}")
    print(f"Test set size: {len(testset)}")

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation set
        pin_memory=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


def load_cifar10(val_ratio, batch_size=128, shuffle=True, num_workers=4):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]
            ),
        ]
    )

    # Load CIFAR-10 training and test datasets
    trainset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # Calculate split sizes
    val_size = int(len(trainset) * val_ratio)
    train_size = len(trainset) - val_size

    # Split training set into train and validation
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

    print(f"Training set size: {len(trainset)}")
    print(f"Validation set size: {len(valset)}")
    print(f"Test set size: {len(testset)}")

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation set
        pin_memory=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


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
            nn.Linear(24 * 8 * 8, 128), nn.BatchNorm1d(128), nn.ReLU()
        )
        self.fc_2 = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU())
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


def train(train_loader, model, optimiser, loss_fn, epochs, device):
    loss_lst = []
    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        running_loss = 0.0
        for x, label in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            # Move data to device non-blocking
            x = x.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            # reset grad
            optimiser.zero_grad()
            label_pred = model(x)
            loss = loss_fn(label_pred, label)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()

        # calcualte average loss per batch for final loss
        avg_loss = running_loss / len(train_loader)
        loss_lst.append(avg_loss)
        # print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")
    return loss_lst


def evaluate(test_loader, model, device):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        metric = MulticlassAccuracy().to(device)  # Move metric to device
        for x, label in tqdm(test_loader):
            # Move data to device non-blocking
            x = x.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            label_pred = model(x)
            metric.update(label_pred, label)

        accuracy = metric.compute()
        print(f"Accuracy: {accuracy:.4f}")


def plot_loss(loss_lst):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_lst) + 1), loss_lst, marker="o")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    # Save the plot
    plt.savefig("training_loss.png")
    plt.close()

    print("Training loss plot saved as 'training_loss.png'")


def main():
    batch_size = 256
    epochs = 20
    learning_rate = 0.01
    val_ratio = 0.2
    num_workers = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Enable cuDNN auto-tuner
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader, test_loader = load_cifar10(
        batch_size, val_ratio, num_workers=num_workers
    )

    model = CustomCNN().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss().to(device)  # Move loss function to device

    loss_lst = train(train_loader, model, optimiser, loss_fn, epochs, device)

    evaluate(test_loader, model, device)

    plot_loss(loss_lst)


if __name__ == "__main__":
    main()
