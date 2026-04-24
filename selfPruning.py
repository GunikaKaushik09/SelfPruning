import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os


# Reproducibility

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Hyperparameters

BATCH_SIZE = 128
EPOCHS = 12
LR = 1e-3

LAMBDA_VALUES = [1e-5, 1e-4, 1e-3]


# CIFAR-10 Data

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)

test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Part 1: Custom PrunableLinear Layer

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # standard weights + bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # learnable gate scores (same shape as weight)
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        # Convert scores to [0,1]
        gates = torch.sigmoid(self.gate_scores)

        # Elementwise prune
        pruned_weights = self.weight * gates

        # Linear transform
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores)



# Neural Network

class SelfPruningNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = PrunableLinear(32*32*3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def sparsity_loss(self):
        total = 0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                total += module.get_gates().sum()
        return total

    def get_all_gates(self):
        gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates.append(module.get_gates().detach().cpu().numpy().flatten())
        return np.concatenate(gates)



# Train Function

def train_model(model, lambda_sparse):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            cls_loss = criterion(outputs, labels)
            sparse_loss = model.sparsity_loss()

            loss = cls_loss + lambda_sparse * sparse_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Lambda={lambda_sparse} | Epoch {epoch+1}/{EPOCHS} | Loss={total_loss:.4f}")


# Test Accuracy

def evaluate(model):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    return acc


# Sparsity %

def calculate_sparsity(model, threshold=1e-2):
    gates = model.get_all_gates()
    zero_like = np.sum(gates < threshold)
    total = len(gates)

    return 100 * zero_like / total


# Run Experiments

results = []
best_model = None
best_acc = 0

for lam in LAMBDA_VALUES:
    print("\nTraining for lambda =", lam)

    model = SelfPruningNet().to(device)

    train_model(model, lam)

    acc = evaluate(model)
    sparsity = calculate_sparsity(model)

    results.append({
        "Lambda": lam,
        "Test Accuracy (%)": round(acc,2),
        "Sparsity (%)": round(sparsity,2)
    })

    if acc > best_acc:
        best_acc = acc
        best_model = model


# Results Table

df = pd.DataFrame(results)
print("\nFinal Results")
print(df)


# Plot Gate Distribution for Best Model

gates = best_model.get_all_gates()

plt.figure(figsize=(8,5))
plt.hist(gates, bins=50, edgecolor='black')
plt.title("Distribution of Final Gate Values (Best Model)")
plt.xlabel("Gate Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()