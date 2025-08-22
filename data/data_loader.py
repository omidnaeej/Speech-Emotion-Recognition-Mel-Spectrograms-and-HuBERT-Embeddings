import os
import re
from glob import glob
from typing import List, Tuple, Dict

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from utils.features import extract_mel_log_db, extract_hubert

import sys
import zipfile
import subprocess
from pathlib import Path

import time

def _find_audio_dir(root: Path, audio_subdir: str) -> Path | None:
    direct = root / audio_subdir
    if direct.exists() and any(direct.glob("*.wav")):
        return direct
    # search nested
    for cand in root.rglob(audio_subdir):
        if any(cand.glob("*.wav")):
            return cand
    return None

def _has_kaggle_creds(dataset_root: str) -> tuple[bool, dict]:
    env = os.environ.copy()
    # prefer data/dataset/kaggle.json if present
    creds_here = Path(dataset_root) / "kaggle.json"
    if creds_here.exists():
        env["KAGGLE_CONFIG_DIR"] = str(Path(dataset_root).resolve())
        return True, env
    # ~/.kaggle/kaggle.json or env vars
    if (Path.home() / ".kaggle" / "kaggle.json").exists() or (
        env.get("KAGGLE_USERNAME") and env.get("KAGGLE_KEY")
    ):
        return True, env
    return False, env

def _extract_zip(zip_path: Path, dst_root: Path) -> None:
    print(f"[CREMA-D] Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst_root)

def ensure_cremad(dataset_root: str, audio_subdir: str = "AudioWAV") -> Path:
    """
    Ensure CREMA-D exists at dataset_root. If not:
      1) If a local zip exists (cremad*.zip), extract it.
      2) Else try Kaggle CLI (with retries).
    Returns path to the AudioWAV folder.
    """
    root = Path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)

    # 0) Already there?
    audio_dir = _find_audio_dir(root, audio_subdir)
    # print(audio_dir)
    if audio_dir:
        return audio_dir

    # 1) If a local zip is present, extract it (fastest path for SSL issues)
    zips = sorted(root.glob("*.zip"), key=lambda p: p.stat().st_size, reverse=True)
    if zips:
        _extract_zip(zips[0], root)
        audio_dir = _find_audio_dir(root, audio_subdir)
        if audio_dir:
            print(f"[CREMA-D] Ready at: {audio_dir.resolve()}")
            return audio_dir
        raise RuntimeError("[CREMA-D] Zip extracted but AudioWAV not found.")

    # 2) Try Kaggle CLI with retries
    has_creds, kaggle_env = _has_kaggle_creds(dataset_root)
    if not has_creds:
        raise RuntimeError(
            "[CREMA-D] Missing dataset and no Kaggle credentials found.\n"
            f"- Put kaggle.json in {root} or ~/.kaggle/, OR\n"
            "- Manually download the dataset zip into this folder as 'cremad.zip'."
        )

    print("[CREMA-D] Not found. Downloading from Kaggle (ejlok1/cremad)...")
    # a few short retries to avoid transient SSL EOF errors
    last_err = None
    for attempt in range(1, 4):
        try:
            cp = subprocess.run(
                ["kaggle", "datasets", "download", "-d", "ejlok1/cremad", "-p", str(root)],
                check=True,
                env=kaggle_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            # identify zip (kaggle may name it differently)
            zips = sorted(root.glob("*.zip"), key=lambda p: p.stat().st_size, reverse=True)
            if not zips:
                raise FileNotFoundError("No zip file found after Kaggle download.")
            _extract_zip(zips[0], root)
            audio_dir = _find_audio_dir(root, audio_subdir)
            if not audio_dir:
                raise RuntimeError("CREMA-D extracted, but 'AudioWAV/*.wav' not found.")
            print(f"[CREMA-D] Ready at: {audio_dir.resolve()}")
            return audio_dir
        except subprocess.CalledProcessError as e:
            last_err = e
            print(f"[CREMA-D] Kaggle download failed (attempt {attempt}/3). Retrying in 5s...")
            time.sleep(5)

    # If all attempts failed, provide a clean escape hatch
    raise RuntimeError(
        "[CREMA-D] Kaggle download kept failing (likely SSL/TLS on this network).\n"
        "➡ Workaround: download the dataset zip in your browser and place it here:\n"
        f"   {root.resolve()}\\cremad.zip\n"
        "Then re-run; the loader will auto-extract."
    )


EMO_MAP = {
    # CREMA-D filename: ID_Sentence_Emotion_... -> Emotion codes
    "NEU": 0,  # Neutral
    "HAP": 1,  # Happy
    "SAD": 2,  # Sad
    "ANG": 3,  # Angry
}

FILENAME_RE = re.compile(r"^(?P<spk>\d{3})_.*_(?P<emo>[A-Z]{3})_.*\.wav$")


def list_wavs(root: str, audio_glob: str) -> List[str]:
    return sorted(glob(os.path.join(root, audio_glob), recursive=True))


def parse_filename(path: str):
    name = os.path.basename(path)
    m = FILENAME_RE.match(name)
    if not m:
        return None
    spk = m.group("spk")
    emo = m.group("emo")
    return spk, emo


class CremadSER(Dataset):
    def __init__(self,
                 wavs: List[str],
                 cfg: dict,
                 feature_type: str = "mel"):
        self.wavs = wavs
        self.cfg = cfg
        self.feature_type = feature_type
        self.sr = cfg["sample_rate"]
        self.clip_len = int(cfg["clip_seconds"] * self.sr)

    def __len__(self):
        return len(self.wavs)

    def _load(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        wav = torch.mean(wav, dim=0, keepdim=True)  # mono [1, T]
        T = wav.shape[1]
        if T < self.clip_len:
            pad = self.clip_len - T
            wav = torch.nn.functional.pad(wav, (0, pad))
        elif T > self.clip_len:
            wav = wav[:, :self.clip_len]
        return wav.squeeze(0)  # [T]

    def __getitem__(self, idx):
        path = self.wavs[idx]
        spk, emo_code = parse_filename(path)
        label = EMO_MAP[emo_code]
        x = self._load(path)
        if self.feature_type == "mel":
            feat = extract_mel_log_db(x, self.sr, self.cfg["mel"])  # [n_mels, time]
        else:
            feat = extract_hubert(x.numpy(), self.sr, self.cfg["hubert"])  # [768]
        return feat, label


def stratified_split(paths: List[str], labels: List[int], val_size: float, test_size: float, seed: int):
    # first split off test, then val from train
    paths = np.array(paths)
    labels = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(paths, labels, test_size=test_size, random_state=seed, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size/(1-test_size), random_state=seed, stratify=y_train)
    return list(X_train), list(y_train), list(X_val), list(y_val), list(X_test), list(y_test)


def build_dataloaders(cfg: dict):
    # NEW: ensure dataset is present (downloads if missing)
    ensure_cremad("./data/dataset")

    all_wavs = list_wavs(cfg["dataset_root"], cfg["audio_glob"])
    print(f"{len(all_wavs)} wavs found.")
    # Filter by selected emotions & limit to N speakers deterministically
    selected = []
    seen_spk = set()
    for p in all_wavs:
        parsed = parse_filename(p)
        if not parsed:
            continue
        spk, emo = parsed
        if emo not in cfg["selected_emotions"]:
            continue
        if len(seen_spk) < cfg["speaker_count"] or spk in seen_spk:
            selected.append(p)
            seen_spk.add(spk)
        if len(seen_spk) >= cfg["speaker_count"]:
            # continue collecting only from these speakers
            pass

    labels = [EMO_MAP[parse_filename(p)[1]] for p in selected]
    X_tr, y_tr, X_va, y_va, X_te, y_te = stratified_split(selected, labels, cfg["val_size"], cfg["test_size"], cfg["random_seed"])

    ds_tr = CremadSER(X_tr, cfg, feature_type=cfg["feature_type"])
    ds_va = CremadSER(X_va, cfg, feature_type=cfg["feature_type"])
    ds_te = CremadSER(X_te, cfg, feature_type=cfg["feature_type"])

    def collate_mel(batch):
        xs, ys = zip(*batch)
        xs = torch.stack(xs)  # [B, n_mels, T]
        ys = torch.tensor(ys, dtype=torch.long)
        return xs, ys

    def collate_hubert(batch):
        xs, ys = zip(*batch)
        xs = torch.stack(xs)  # [B, 768]
        ys = torch.tensor(ys, dtype=torch.long)
        return xs, ys

    is_mel = cfg["feature_type"] == "mel"
    collate_fn = collate_mel if is_mel else collate_hubert

    dl_tr = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"], collate_fn=collate_fn)
    dl_va = DataLoader(ds_va, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"], collate_fn=collate_fn)
    dl_te = DataLoader(ds_te, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"], collate_fn=collate_fn)

    return dl_tr, dl_va, dl_te