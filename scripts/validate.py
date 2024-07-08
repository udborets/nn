from typing import Callable, TypedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ValResults(TypedDict):
    acc: float
    loss: float


def validate(
    model: nn.Module,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    loader: DataLoader,
    device: torch.device = torch.device("cuda"),
) -> ValResults:
    model.to(device).eval()
    criterion.to(device).eval()
    n_batches = len(loader)
    running_loss = 0.0
    running_acc = 0.0
    with torch.inference_mode():
        for images, labels in loader:
            images.to(device)
            labels.to(device)
            out: torch.Tensor = model(images)
            loss = criterion(out, labels)
            running_loss += loss.detach().item()
            running_acc += out.argmax(1).eq(labels).item() / len(labels)
    running_loss /= n_batches
    running_acc /= n_batches
    res: ValResults = {
        "acc": running_acc,
        "loss": running_loss,
    }
    return res
