import os
from typing import Dict, List
import matplotlib.pyplot as plt


def plot_history(history: Dict[str, List[float]], outdir: str, title: str = "Training"):
    os.makedirs(outdir, exist_ok=True)
    # Accuracy
    plt.figure()
    plt.plot(history["train_acc"], label="train")
    plt.plot(history["val_acc"], label="val")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title(f"{title} — Accuracy"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "acc.png")); plt.close()
    # Loss
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title(f"{title} — Loss"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "loss.png")); plt.close()
    