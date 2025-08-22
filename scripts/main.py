import yaml
from scripts.train import fit
from scripts.evaluate import evaluate_test
from utils.visualization import plot_history
import torch


def main():
    with open("./config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    model, (dl_tr, dl_va, dl_te), history, ckpt_path = fit(cfg)
    plot_history(history, outdir=cfg.get("log_dir", "./runs/ser"), title=f"Training ({cfg['feature_type']})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc, cm, report = evaluate_test(model, dl_te, device)
    print("\nTest accuracy:", acc)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)

if __name__ == "__main__":
    main()