from tqdm import tqdm
import os
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torcheval.metrics import MulticlassAccuracy
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = "results/"


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
        persistent_workers=True,
    )

    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation set
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
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
            nn.Dropout(0.3),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3),
        )
        self.fc_1 = nn.Sequential(
            nn.Linear(24 * 8 * 8, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3)
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


class SimpleNet(torch.nn.Module):
    def __init__(
        self, model_dim=1024, num_layers=4, dropout=0.3, activation_function="relu"
    ):
        super(SimpleNet, self).__init__()
        self.fc_layers = nn.ModuleList()

        # create a list of linear layers
        self.fc_input = nn.Sequential(
            nn.Linear(32 * 32 * 3, model_dim),
            nn.BatchNorm1d(model_dim),
            get_activation_function(activation_function),
            nn.Dropout(dropout),
        )
        for _ in range(num_layers):
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(model_dim, model_dim),
                    nn.BatchNorm1d(model_dim),
                    get_activation_function(activation_function),
                    nn.Dropout(dropout),
                )
            )

        self.fc_final = nn.Sequential(
            nn.Linear(model_dim, 10),
        )

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = self.fc_input(x)
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        x = self.fc_final(x)
        return x


def evaluate(data_loader, model, loss_fn, device):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        running_loss = 0.0
        metric = MulticlassAccuracy().to(device)  # Move metric to device
        for x, label in tqdm(data_loader):
            # Move data to device non-blocking
            x = x.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            label_pred = model(x)
            loss = loss_fn(label_pred, label)
            running_loss += loss.item()  # Convert loss to float immediately
            metric.update(label_pred, label)

        accuracy = metric.compute()
        avg_loss = running_loss / len(data_loader)

        logger.info(
            f"Evaluation - Accuracy: {accuracy.item():.4f}, Loss: {avg_loss:.4f}"
        )
        return accuracy.item(), avg_loss  # Convert accuracy to float


def train(
    train_loader,
    vald_loader,
    model,
    optimiser,
    loss_fn,
    epochs,
    device,
    lr_scheduler=None,
):
    train_Loss = []
    val_accuracy = []
    val_loss = []
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

        # calculate average loss per batch for final loss
        avg_loss = running_loss / len(train_loader)
        train_Loss.append(avg_loss)
        logger.info(f"Epoch {epoch + 1} Training Loss: {avg_loss:.4f}")

        # evaluate on validation set
        eval_accuracy, eval_loss = evaluate(vald_loader, model, loss_fn, device)
        val_accuracy.append(eval_accuracy)  # Already converted to float in evaluate
        val_loss.append(eval_loss)

        if lr_scheduler:
            lr_scheduler.step(eval_loss)

        logger.info(
            f"Epoch {epoch + 1} - Train Loss: {avg_loss:.4f}, Val Loss: {eval_loss:.4f}, Val Acc: {eval_accuracy:.4f}"
        )
    return train_Loss, val_accuracy, val_loss


def plot_loss(train_Loss, val_loss):
    # Force conversion to make sure we have Python floats
    train_Loss = [float(x) if torch.is_tensor(x) else float(x) for x in train_Loss]
    val_loss = [float(x) if torch.is_tensor(x) else float(x) for x in val_loss]

    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(train_Loss) + 1), train_Loss, marker="o", label="Training Loss"
    )
    plt.plot(range(1, len(val_loss) + 1), val_loss, marker="o", label="Validation Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Save the plot to RESULTS_DIR
    os.makedirs(RESULTS_DIR, exist_ok=True)  # Create directory if it doesn't exist
    plt.savefig(RESULTS_DIR + "training_and_validation_loss.png")
    plt.close()

    print(
        f"Training and validation loss plot saved as '{RESULTS_DIR}training_and_validation_loss.png'"
    )


def plot_accuracy(accuracy):
    # Force conversion to make sure we have Python floats
    accuracy = [float(x) if torch.is_tensor(x) else float(x) for x in accuracy]

    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(accuracy) + 1), accuracy, marker="o", label="Validation Accuracy"
    )
    plt.title("Validation Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")  # Updated to show it's percentage
    plt.legend()
    plt.grid(True)

    # Save the plot to RESULTS_DIR
    os.makedirs(RESULTS_DIR, exist_ok=True)  # Create directory if it doesn't exist
    plt.savefig(RESULTS_DIR + "validation_accuracy.png")
    plt.close()

    print(f"Validation accuracy plot saved as '{RESULTS_DIR}validation_accuracy.png'")


def get_activation_function(activation_function):
    if activation_function == "relu":
        return nn.ReLU()
    elif activation_function == "gelu":
        return nn.GELU()
    elif activation_function == "elu":
        return nn.ELU()
    else:
        raise ValueError(f"Activation function {activation_function} not supported")


def main():
    # Enable cuDNN auto-tuner
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True
    torch.set_float32_matmul_precision("high")

    val_ratio = 0.2
    num_workers = 8
    # training parameters
    batch_size = 128
    epochs = 200
    learning_rate = 0.0001
    weight_decay = 0.000001
    lr_scheduler_patience = 10
    lr_scheduler_factor = 0.2
    seed = 42
    # model
    model_dim = 128
    num_layers = 64
    dropout = 0.2
    activation_function = "elu"

    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = load_cifar10(
        val_ratio=val_ratio, batch_size=batch_size, num_workers=num_workers
    )

    model = SimpleNet(
        model_dim=model_dim,
        num_layers=num_layers,
        dropout=dropout,
        activation_function=activation_function,
    ).to(device)

    # Compile the model for better performance
    try:
        model = torch.compile(model)
        print("Model successfully compiled with torch.compile()")
    except Exception as e:
        print(f"Could not compile model: {e}. Continuing with uncompiled model.")

    print(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1000:.1f}k")

    optimiser = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss().to(device)  # Move loss function to device
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
    )

    train_Loss, val_accuracy, val_loss = train(
        train_loader,
        val_loader,
        model,
        optimiser,
        loss_fn,
        epochs,
        device,
        lr_scheduler=lr_scheduler,
    )

    accuracy, avg_loss = evaluate(test_loader, model, loss_fn, device)

    print(50 * "-")
    print(f"Final Val Accuracy: {val_accuracy[-1] * 100:.1f}%")
    print(f"Final Val Loss: {val_loss[-1]:.4f}")
    print()
    print(f"Test Accuracy: {accuracy * 100:.1f}%")
    print(f"Average Test Loss: {avg_loss:.4f}")

    plot_loss(train_Loss, val_loss)
    plot_accuracy(val_accuracy)


if __name__ == "__main__":
    main()
