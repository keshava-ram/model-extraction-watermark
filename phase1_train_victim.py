"""
PHASE 1 — Train the Victim Model
=================================
Model Extraction Attack & Watermark Defense Project

What this script does:
  1. Downloads CIFAR-10 (10-class image dataset, 32x32 color images)
  2. Defines a small CNN
  3. Trains it for a number of epochs, printing train/test accuracy each epoch
  4. Saves the trained weights to victim_model.pt

HOW TO RUN THIS (recommended: Google Colab, free GPU, nothing to install):
  1. Go to https://colab.research.google.com -> New Notebook
  2. Runtime -> Change runtime type -> select "GPU" (T4 is fine)
  3. Paste this whole file into a cell and run it
  4. After it finishes, victim_model.pt will appear in Colab's file panel (left sidebar)
     -> right-click it -> Download, to keep a local copy

If running locally instead: `pip install torch torchvision` first.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ----------------------------------------------------------------------
# 0. Setup
# ----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

EPOCHS = 20          # ~15-20 epochs is enough to reach ~85-90% test accuracy on CIFAR-10
BATCH_SIZE = 128
LEARNING_RATE = 0.001

# ----------------------------------------------------------------------
# 1. Data: CIFAR-10 (10 classes: plane, car, bird, cat, deer, dog, frog,
#    horse, ship, truck — 50,000 train images, 10,000 test images)
# ----------------------------------------------------------------------
# Normalization values are the standard, well-known CIFAR-10 channel mean/std
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),   # light data augmentation
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ----------------------------------------------------------------------
# 2. Model: a small CNN (3 conv blocks + 2 fully-connected layers)
#    This is deliberately simple -- enough to reach solid accuracy
#    without needing a huge, slow-to-train architecture.
# ----------------------------------------------------------------------
class VictimCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 32x32x3 -> 16x16x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 16x16x32 -> 8x8x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 8x8x64 -> 4x4x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = VictimCNN(num_classes=10).to(device)
print(model)

# ----------------------------------------------------------------------
# 3. Loss, optimizer
# ----------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# ----------------------------------------------------------------------
# 4. Evaluation helper
# ----------------------------------------------------------------------
def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


# ----------------------------------------------------------------------
# 5. Training loop
# ----------------------------------------------------------------------
print("\nStarting training...\n")
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    scheduler.step()
    train_loss = running_loss / total
    train_acc = 100.0 * correct / total
    test_acc = evaluate(test_loader)

    print(f"Epoch {epoch:2d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:5.2f}% | "
          f"Test Acc: {test_acc:5.2f}%")

# ----------------------------------------------------------------------
# 6. Save the trained victim model
# ----------------------------------------------------------------------
torch.save(model.state_dict(), "victim_model.pt")
final_acc = evaluate(test_loader)
print(f"\nDone. Final test accuracy: {final_acc:.2f}%")
print("Saved weights to victim_model.pt")
