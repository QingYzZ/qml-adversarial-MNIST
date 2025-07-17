import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from MNIST.LewHybridNN import LewHybridNN
from MNIST.ClassicalCNN import ClassicalCNN
import random


# 🧪 Custom: Add Gaussian Noise
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.3):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


# 🧪 Custom: Add Occlusion Mask
class AddOcclusion(object):
    def __init__(self, size=8):
        self.size = size

    def __call__(self, tensor):
        _, h, w = tensor.shape
        x = random.randint(0, w - self.size)
        y = random.randint(0, h - self.size)
        tensor[:, y:y+self.size, x:x+self.size] = 0
        return tensor


# 🧠 Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_q = LewHybridNN().to(device)
model_q.load_state_dict(torch.load("hybrid_qnn.pth"))
model_q.eval()

model_c = ClassicalCNN().to(device)
model_c.load_state_dict(torch.load("classical_cnn.pth"))
model_c.eval()

# 📊 Evaluation function
def evaluate(model, loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total


# 🧪 Define corruptions
corruptions = {
    "Original": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    "Gaussian Noise": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        AddGaussianNoise(0., 0.3)
    ]),
    "Rotation ±30°": transforms.Compose([
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    "Contrast Change": transforms.Compose([
        transforms.ColorJitter(contrast=2.0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    "Occlusion": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        AddOcclusion(size=8)
    ]),
}


# 🧪 Run Evaluation Across All Corruptions
hybrid_accs = []
classical_accs = []

for name, transform in corruptions.items():
    print(f"\n🔍 Evaluating on: {name}")
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    acc_q = evaluate(model_q, test_loader)
    acc_c = evaluate(model_c, test_loader)

    hybrid_accs.append(acc_q * 100)
    classical_accs.append(acc_c * 100)

    print(f"Hybrid QNN Accuracy: {acc_q * 100:.2f}%")
    print(f"Classical CNN Accuracy: {acc_c * 100:.2f}%")


# 📊 Final Bar Chart
labels = list(corruptions.keys())
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(12, 6))
bars1 = plt.bar(x - width/2, hybrid_accs, width, label='Hybrid QNN', color='skyblue')
bars2 = plt.bar(x + width/2, classical_accs, width, label='Classical CNN', color='lightgreen')

for bar, acc in zip(bars1, hybrid_accs):
    yval = bar.get_height()
    offset = -5 if yval > 95 else 1  # Show below bar if close to 100%
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + offset, f'{acc:.1f}%', ha='center', va='bottom')

for bar, acc in zip(bars2, classical_accs):
    yval = bar.get_height()
    offset = -5 if yval > 95 else 1
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + offset, f'{acc:.1f}%', ha='center', va='bottom')


plt.xticks(x, labels, rotation=15)
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy under Different Input Corruptions')
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig("corruption_comparison_bar_chart.png", dpi=300)
plt.show()
