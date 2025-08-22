import os
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from tqdm import tqdm

from models.model import build_model
from data.data_loader import build_dataloaders
from utils.metrics import accuracy


def get_optimizer(name, params, lr, weight_decay=0.0):
    name = name.lower()
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer {name}")


def train_one_epoch(model, dl, device, criterion, optimizer):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for x, y in tqdm(dl, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            running_loss += loss.item() * y.size(0)
            running_acc  += accuracy(logits, y) * y.size(0)
            n += y.size(0)
    return running_loss / n, running_acc / n


def evaluate(model, dl, device, criterion):
    model.eval()
    running_loss, running_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in tqdm(dl, desc="eval", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * y.size(0)
            running_acc  += accuracy(logits, y) * y.size(0)
            n += y.size(0)
    return running_loss / n, running_acc / n


def fit(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl_tr, dl_va, dl_te = build_dataloaders(cfg)

    model = build_model(cfg["feature_type"], cfg)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    opt = get_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))

    best_val = 0.0
    ckpt_path = os.path.join(cfg["ckpt_dir"], cfg["ckpt_name"].replace("${feature_type}", cfg["feature_type"]))
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(model, dl_tr, device, criterion, opt)
        va_loss, va_acc = evaluate(model, dl_va, device, criterion)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        print(f"Epoch {epoch:02d}: train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={va_loss:.4f} acc={va_acc:.4f}")
        if va_acc > best_val:
            best_val = va_acc
            torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)
            print(f"Saved best model to {ckpt_path}")

    return model, (dl_tr, dl_va, dl_te), history, ckpt_path