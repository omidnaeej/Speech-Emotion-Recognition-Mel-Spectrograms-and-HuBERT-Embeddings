import torch
import yaml
import numpy as np
import os
from data.data_loader import build_dataloaders
from scripts.train import fit
from utils.metrics import compute_metrics
from utils.visualization import plot_history

import warnings
warnings.filterwarnings("ignore")

def main():
    with open("./config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    model, (dl_tr, dl_va, dl_te), history, ckpt_path = fit(cfg)
    plot_history(history, outdir=cfg.get("log_dir", "./runs/ser"), title=f"Training ({cfg['feature_type']})")
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dl_te:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_labels, all_preds, outdir=cfg.get("log_dir", "./runs/ser"))
    print(f"Test accuracy: {metrics['accuracy']}")
    print("Confusion Matrix:\n", metrics["confusion_matrix"])
    print("Classification Report:\n", yaml.dump(metrics["classification_report"], default_flow_style=False))

if __name__ == "__main__":
    main()