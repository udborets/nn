from typing import Callable
from torch.utils.data import DataLoader
import torch
import torch.nn as nn


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    loader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device = torch.device("cuda"),
):
    n_batches = len(loader)
    for epoch in range(num_epochs):
        model.to(device).train()
        loss.to(device).train()
        running_loss = 0.0
        print(f"Epoch {epoch + 1}")
        for images, labels in loader:
            images.to(device)
            labels.to(device)
            out = model(images)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.detach().item()
            running_loss += loss
            
        running_loss /= n_batches
