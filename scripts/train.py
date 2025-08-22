import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

from models.model import build_model
from utils.metrics import compute_metrics
from data.data_loader import collate_mel, collate_hubert, build_dataloaders

def train_one_epoch(model: nn.Module, dl: DataLoader, device: torch.device, criterion, opt):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for x, y in tqdm(dl, desc="train", leave=True):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = np.mean(all_preds == all_labels)
    return total_loss / len(dl), acc

def eval_one_epoch(model: nn.Module, dl: DataLoader, device: torch.device, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for x, y in tqdm(dl, desc="eval", leave=True):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = np.mean(all_preds == all_labels)
    return total_loss / len(dl), acc

def fit(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    model = model.to(device)
    dl_tr, dl_va, dl_te = build_dataloaders(cfg)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]) if cfg["optimizer"] == "adam" else torch.optim.SGD(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    ckpt_dir = cfg["ckpt_dir"]
    ckpt_path = f"{ckpt_dir}/{cfg['ckpt_name'].replace('${feature_type}', cfg['feature_type'])}"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val_acc = -float("inf")
    for epoch in range(cfg["epochs"]):
        tr_loss, tr_acc = train_one_epoch(model, dl_tr, device, criterion, opt)
        val_loss, val_acc = eval_one_epoch(model, dl_va, device, criterion)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch+1:02d}: train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"✅ Saved best model to {ckpt_path}")
    return model, (dl_tr, dl_va, dl_te), history, ckpt_path