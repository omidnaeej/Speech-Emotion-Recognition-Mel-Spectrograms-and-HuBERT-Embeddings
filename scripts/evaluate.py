import torch
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from utils.metrics import accuracy


def evaluate_test(model, dl_te, device):
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for x, y in dl_te:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_p.append(preds)
            all_y.append(y.numpy())
    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    cm = confusion_matrix(y, p)
    report = classification_report(y, p, digits=4)
    acc = (y == p).mean()
    return acc, cm, report