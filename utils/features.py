from typing import Dict
import torch
import torchaudio

# ===== Log-Mel spectrogram (then log-scale) =====

def extract_mel_log_db(wav: torch.Tensor, sr: int, mcfg: Dict) -> torch.Tensor:
    """wav: [T] float32 in [-1,1]; returns [n_mels, time] log-mel (dB)."""
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=mcfg.get("n_fft", 1024),
        hop_length=mcfg.get("hop_length", 320),
        win_length=mcfg.get("win_length", 640),
        f_min=mcfg.get("f_min", 0.0),
        f_max=mcfg.get("f_max", sr//2),
        n_mels=mcfg.get("n_mels", 64),
        center=mcfg.get("center", True),
        power=2.0,
        norm=None,
        mel_scale="htk",
    )(wav)
    log_mel = torchaudio.transforms.AmplitudeToDB(stype="power")(mel_spec)
    return log_mel  # [n_mels, time]

# ===== HuBERT embeddings (pooled) =====

# def extract_hubert(wav_np, sr: int, hcfg: Dict) -> torch.Tensor:
#     from transformers import Wav2Vec2FeatureExtractor, HubertModel
#     import numpy as np

#     if sr != 16000:
#         # Expect 16k — resample using torchaudio for consistency
#         wav_t = torch.tensor(wav_np, dtype=torch.float32)
#         wav_t = torchaudio.functional.resample(wav_t, sr, 16000)
#         wav_np = wav_t.numpy()
#         sr = 16000

#     extractor = Wav2Vec2FeatureExtractor.from_pretrained(hcfg.get("model_name", "facebook/hubert-base-ls960"))
#     model = HubertModel.from_pretrained(hcfg.get("model_name", "facebook/hubert-base-ls960"))
#     model.eval()
#     with torch.inference_mode():
#         inputs = extractor(wav_np, sampling_rate=sr, return_tensors="pt")
#         out = model(**inputs)
#         hidden = out.last_hidden_state  # [1, T, 768]
#         if hcfg.get("layer_pooling", "mean") == "cls":
#             feat = hidden[:, 0, :]
#         else:
#             feat = hidden.mean(dim=1)  # [1, 768]
#     return feat.squeeze(0)  # [768]