from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.optimizer import ParamsT

class Lion(torch.optim.Optimizer):
    def __init__(
            self,
            params: ParamsT,
            lr: float,
            betas: Tuple[float]=(0.9, 0.999),
            weight_decay:float = 0.0):

        defaults = {
            "params":  params,
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay
        }

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad


                state = self.state[p]

                if "m" not in state:
                    state["m"] = torch.zeros_like(p)

                update = torch.sign(state["m"] * beta1 + (1 - beta1) * grad)

                if weight_decay != 0.0:
                    update = update + weight_decay * p

                update_by = update * lr

                p.add_(-update_by)

                state["m"] = state["m"] * beta2 + (1 - beta2) * grad


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)

    for epoch in range(20):
        model.train()
        running_loss = 0.0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.4f}")


# Epoch 1, Loss: 1.8090
# Epoch 2, Loss: 1.4175
# Epoch 3, Loss: 1.2480
# Epoch 4, Loss: 1.1169
# Epoch 5, Loss: 1.0111
# Epoch 6, Loss: 0.9248
# Epoch 7, Loss: 0.8496
# Epoch 8, Loss: 0.7834
# Epoch 9, Loss: 0.7230
# Epoch 10, Loss: 0.6676
# Epoch 11, Loss: 0.6149
# Epoch 12, Loss: 0.5639
# Epoch 13, Loss: 0.5146
# Epoch 14, Loss: 0.4674
# Epoch 15, Loss: 0.4211
# Epoch 16, Loss: 0.3755
# Epoch 17, Loss: 0.3297
# Epoch 18, Loss: 0.2867
# Epoch 19, Loss: 0.2422
# Epoch 20, Loss: 0.2008