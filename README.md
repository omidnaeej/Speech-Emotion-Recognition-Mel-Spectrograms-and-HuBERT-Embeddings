# Speech Emotion Recognition using Log-Mel Spectrograms & HuBERT Embeddings

This repository implements Speech Emotion Recognition (SER) using **log-Mel spectrograms** and **HuBERT embeddings** on the **CREMA-D** dataset. It includes data loading, feature extraction, compact models for both feature types, training/evaluation scripts, and result visualizations (accuracy, loss, confusion matrix). ([GitHub][1])

---

## Repository Contents

* `config/` – YAML configs for data, features, training and logging (e.g., `config.yaml`, `logging.yaml`). 
* `data/` – Dataset utilities (e.g., `data_loader.py` with `CremadSER` and helpers).
* `models/` – Model definitions & saved checkpoints (e.g., `model.py`, `models/saved_models/ser_mel.pt`, `ser_hubert.pt`).
* `runs/` – Training logs and plots (`acc.png`, `loss.png`, `cm.png`) for Mel and HuBERT experiments. 
* `scripts/` – Entry points (`main.py`) and helpers (`train.py`, `evaluate.py`). 
* `utils/` – Feature extraction (`features.py`), metrics, and visualization utilities. 
* `requirements.txt`, `README.md`, `.gitignore`.

---

## Project Overview

#### 1) Data Preparation

* Target dataset: **CREMA-D**; files reside under `AudioWAV/`. Set `dataset_root` to the folder that **contains** `AudioWAV`. Example default: `./data/dataset/AudioWAV`.
* Emotions used: **NEU, HAP, SAD, ANG** (mapped to 0–3). Fixed sample rate **16 kHz** and clips **3 s**
* The data loader normalizes mono audio, resamples if needed, trims/pads to fixed length, and yields either Mel or HuBERT features depending on config. 

#### 2) Feature Extraction

* **Log-Mel**: Configurable STFT params; default `n_mels: 64`, `n_fft: 1024`, `hop_length: 320`, `win_length: 640`, `f_max: 8000`. 
* **HuBERT**: Uses `facebook/hubert-base-ls960`; embeddings pooled by `mean` (configurable).
* The loader calls `utils.features.extract_mel_log_db` and `extract_hubert` under the hood. 

#### 3) Model Architecture

* **Mel branch**: small **CNN** classifier; default channels `[32, 64, 128]`, dropout `0.2`, `num_classes: 4`.
* **HuBERT branch**: **MLP** over pooled 768-d HuBERT embeddings; default `hidden_dims: [256]`, `num_classes: 4`. (A later config also experimented with `[512, 128]`.) 

#### 4) Training & Evaluation

* Default split: `val_size: 0.1`, `test_size: 0.1`, `random_seed: 42`.
* Typical training hyper-params: `batch_size: 128`, `epochs: 20`, `optimizer: adam`, `lr: 0.01`. (Experiments with `lr=0.001` are noted in commit history.) 
* Checkpoints saved as `models/saved_models/ser_${feature_type}.pt`; logs/plots under `runs/ser_${feature_type}`.

---

## Setup

Clone:

```bash
git clone https://github.com/omidnaeej/Speech-Emotion-Recognition-Mel-Spectrograms-and-HuBERT-Embeddings.git
cd Speech-Emotion-Recognition-Mel-Spectrograms-and-HuBERT-Embeddings
```

Install dependencies:

```bash
pip install -r requirements.txt
# or
pip install torch torchaudio transformers scikit-learn matplotlib pyyaml tqdm
```

---

## Download the dataset

1. Obtain **CREMA-D** (you’ll need access approval) and place/extract it so that the audio files live in `.../AudioWAV/*.wav`. The default config expects `./data/dataset/AudioWAV`. 
2. Update `config/config.yaml` → `dataset_root` if you put it elsewhere. 

---

## Usage

```bash
python -m scripts.main
```

This reads `config/config.yaml`, builds the selected feature pipeline (`feature_type: "mel"` or `"hubert"`), trains, evaluates, and writes plots/checkpoints. 

Alternative script entry points (present in `scripts/`):

* `scripts/train.py` – training logic
* `scripts/evaluate.py` – evaluation & plotting

> Use `config/config.yaml` to switch features and hyper-parameters (see next section).

---

## Configuration

Edit **`config/config.yaml`** to control the run:

* **Data**

  * `dataset_root`: path to folder containing `AudioWAV`
  * `selected_emotions`: `["NEU","HAP","SAD","ANG"]`
  * `sample_rate`: `16000`, `clip_seconds`: `3`
* **Features**

  * `feature_type`: `"mel"` or `"hubert"`
  * `mel`: `n_fft`, `hop_length`, `win_length`, `n_mels`, `f_max`, ...
  * `hubert`: `model_name` (default `"facebook/hubert-base-ls960"`), `layer_pooling` (`"mean"`/`"cls"`)
* **Split**

  * `val_size`, `test_size`, `random_seed`, `stratify_on`
* **Training**

  * `batch_size`, `epochs`, `lr`, `optimizer`, `weight_decay`, `num_workers`, `pin_memory`
* **Model**

  * `mel_model`: `type: "cnn"`, `channels`, `dropout`, `num_classes`
  * `hubert_model`: `type: "mlp"`, `hidden_dims`, `num_classes`
* **Logging / Checkpoints**

  * `log_dir`: e.g., `./runs/ser_${feature_type}`
  * `ckpt_dir`, `ckpt_name`: e.g., `ser_${feature_type}.pt`

---

## Results

After training, you should see:

* **Plots** under `runs/ser_mel` or `runs/ser_hubert`: `acc.png`, `loss.png`, `cm.png` (confusion matrix). 
* **Checkpoints** saved to `models/saved_models/ser_mel.pt` and `ser_hubert.pt` (depending on the run).

---

## Contributing

Issues and PRs are welcome. Please keep configs, scripts, and utils consistent with the current structure and logging/checkpoint conventions.


[4]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4313618/?utm_source=chatgpt.com "CREMA-D: Crowd-sourced Emotional Multimodal Actors ..."

 
