# MNIST/training.py

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from MNIST.LewHybridNN import LewHybridNN

# Setup
batch_size = 4
num_train = 8000
num_epochs = 3
learning_rate = 0.001


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_dataset, _ = random_split(full_dataset, [num_train, len(full_dataset) - num_train])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model
model = LewHybridNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for name, param in model.named_parameters():
    print(name, param.requires_grad)

# Training
loss_list = []

for epoch in range(num_epochs):
    print(f"\n🔁 Epoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for step, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)

        running_loss += loss.item()
        running_correct += correct
        running_total += total

        if (step + 1) % 10 == 0 or step == 0:
            print(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss.item():.4f}, Accuracy: {correct/total*100:.2f}%")

    avg_loss = running_loss / len(train_loader)
    avg_acc = running_correct / running_total
    print(f"\n📊 Epoch {epoch+1} Summary — Avg Loss: {avg_loss:.4f}, Accuracy: {avg_acc*100:.2f}%")

# Plot loss
plt.figure(figsize=(8, 5))
plt.plot(loss_list, marker='o')
plt.title('Training Loss')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss.png", dpi=300)
plt.show()
