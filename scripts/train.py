from typing import Callable
from torch.utils.data import DataLoader
import torch
import torch.nn as nn


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    loader_train: DataLoader,
    loader_val: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device = torch.device("cuda"),
    loader_test = None,
):
    for epoch in range(num_epochs):
        model.to(device).train()
        loss.to(device).train()
        running_loss = 0.0
        print(f"Epoch {epoch + 1}")
        for images, labels in loader_train:
            images.to(device)
            labels.to(device)
            out = model(images)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.detach().item()
            running_loss += loss
            
        running_loss /= len(loader_train)
