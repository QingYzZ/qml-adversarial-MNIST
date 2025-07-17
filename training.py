import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from MNIST.LewHybridNN import LewHybridNN
from MNIST.ClassicalCNN import ClassicalCNN
import numpy as np

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

# Training function
def train_model(model, train_loader, device, optimizer, criterion, num_epochs):
    model = model.to(device)
    loss_list = []
    acc_list = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)
            running_loss += loss.item()

        loss_list.append(running_loss / len(train_loader))
        acc_list.append(running_correct / running_total)

        print(f"Epoch {epoch+1}/{num_epochs} — Loss: {loss_list[-1]:.4f}, Accuracy: {acc_list[-1]*100:.2f}%")

    return loss_list, acc_list

# Criterion
criterion = nn.CrossEntropyLoss()

# Model 1: Quantum Hybrid
print("Training Hybrid Quantum-Classical Model")
model_q = LewHybridNN()
optimizer_q = optim.Adam(model_q.parameters(), lr=learning_rate)
loss_q, acc_q = train_model(model_q, train_loader, device, optimizer_q, criterion, num_epochs)

# Model 2: Classical CNN
print("Training Classical CNN Model")
model_c = ClassicalCNN()
optimizer_c = optim.Adam(model_c.parameters(), lr=learning_rate)
loss_c, acc_c = train_model(model_c, train_loader, device, optimizer_c, criterion, num_epochs)

# Plot results
epochs = np.arange(1, num_epochs + 1)

plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(epochs, loss_q, label='Hybrid QNN', marker='o')
plt.plot(epochs, loss_c, label='Classical CNN', marker='s')
plt.title('Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(epochs, [a * 100 for a in acc_q], label='Hybrid QNN', marker='o')
plt.plot(epochs, [a * 100 for a in acc_c], label='Classical CNN', marker='s')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("comparison_plot.png", dpi=300)
plt.show()

# Saving Models
torch.save(model_q.state_dict(), "hybrid_qnn.pth")
torch.save(model_c.state_dict(), "classical_cnn.pth")
