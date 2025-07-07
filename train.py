import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from MNIST.dataloader import get_mnist_loaders
from MNIST.model import SimpleQNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load data
train_loader, test_loader = get_mnist_loaders(batch_size=4, num_train=8000)  # smaller batch for speed/debugging

# Model + loss + optimizer
model = SimpleQNN(n_wires=16).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 3
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

        # Track accuracy
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total

        running_loss += loss.item()
        running_correct += correct
        running_total += total

        if (step + 1) % 10 == 0 or step == 0:
            print(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss.item():.4f}, Accuracy: {accuracy*100:.2f}%")

    # print(f"\n✅ Epoch {epoch+1} complete. Last batch loss: {loss.item():.4f}")
    avg_loss = running_loss / len(train_loader)
    avg_acc = running_correct / running_total
    print(f"\n📊 Epoch {epoch+1} Summary — Avg Loss: {avg_loss:.4f}, Accuracy: {avg_acc*100:.2f}%")


# 🔍 Plot loss after training
plt.figure(figsize=(8, 5))
plt.plot(loss_list, marker='o')
plt.title('Training Loss')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss.png", dpi=300)
plt.show()
