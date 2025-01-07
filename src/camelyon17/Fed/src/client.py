from typing import Type
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def validate_step(
    model,
    loss_fn,
    validate_dataloader,
    device):

    model.eval()

    validate_loss, validate_acc = 0.0,0.0
    
    with torch.inference_mode():

        for batch, (x,y) in enumerate(validate_dataloader):
            x, y = x.to(device), y.to(device)

            y_pred_logits = model(x)

            loss = loss_fn(y_pred_logits, y)
            validate_loss += loss.item()

            y_pred_labels = torch.argmax(torch.softmax(y_pred_logits, dim=1), dim=1)
            validate_acc += (y_pred_labels == y).sum().item()/len(y_pred_labels)

    
    validate_loss /= len(validate_dataloader)
    validate_acc /= len(validate_dataloader)


    return validate_loss, validate_acc

def test_step(
    model,
    loss_fn,
    test_dataloader,
    device):

    model.eval()

    test_loss, test_acc = 0.0,0.0
    
    with torch.inference_mode():

        for batch, (x,y) in enumerate(test_dataloader):
            x, y = x.to(device), y.to(device)

            y_pred_logits = model(x)

            loss = loss_fn(y_pred_logits, y)
            test_loss += loss.item()

            y_pred_labels = torch.argmax(torch.softmax(y_pred_logits, dim=1), dim=1)
            test_acc += (y_pred_labels == y).sum().item()/len(y_pred_labels)

    
    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)


    return test_loss, test_acc


def train_step(
    model,
    optimizer,
    loss_fn,
    train_dataloader,
    device    
    ):
    train_loss, train_acc = 0.0, 0.0
    for batch, (x,y) in enumerate(train_dataloader):
        x,y = x.to(device), y.to(device)
        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim = 1), dim = 1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
    
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    
    return train_loss, train_acc



def train_client(
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: Type[DataLoader],
        validate_dataloader: Type[DataLoader],
        device,
        epochs,
        writer = None,):

    results = {"train_loss": [],
            "train_acc": [],
            "validate_loss": [],
            "validate_acc": []
    }

    for epoch in range(epochs):
        train_loss, train_acc = train_step(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_dataloader=train_dataloader,
            device=device
        )

        validate_loss, validate_acc = validate_step(
            model=model,
            validate_dataloader=validate_dataloader,
            loss_fn=loss_fn,
            device=device
        )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["validate_loss"].append(validate_loss)
        results["validate_acc"].append(validate_acc)


    return results, model