# Speech Emotion Recognition using Log-Mel Spectrograms & HuBERT Embeddings

This repository implements a pipeline for **speech emotion recognition (SER)** on the **CREMA-D** dataset using two complementary feature types:

1. **Log-Mel spectrograms** (handcrafted / acoustic features)  
2. **HuBERT embeddings** (self-supervised representations from a pretrained transformer model)

It includes training, evaluation, result plotting (e.g. confusion matrices), and utilities.

---

## 📋 Table of Contents

- [Motivation](#motivation)  
- [Features & Highlights](#features--highlights)  
- [Repository Structure](#repository-structure)  
- [Setup & Requirements](#setup--requirements)  
- [Usage](#usage)  
  - [Data Preparation](#data-preparation)  
  - [Training](#training)  
  - [Evaluation & Inference](#evaluation--inference)  
  - [Visualization & Analysis](#visualization--analysis)  
- [Configuration](#configuration)  
- [Results & Metrics](#results--metrics)  
- [Extending / Customizing](#extending--customizing)  
- [Citations & References](#citations--references)  
- [License](#license)

---

## Motivation

Emotion detection in speech is a challenging problem. Traditional acoustic features (e.g. spectrograms, MFCCs) capture low-level information, while modern self-supervised models like HuBERT can extract richer, semantically meaningful representations. This repository explores combining both and evaluating their individual and fused performance.

---

## Features & Highlights

- Support for **log-Mel spectrograms** and **HuBERT embeddings**  
- Training and evaluation scripts  
- Automatic generation of confusion matrices and plots  
- Modular utilities for loading, preprocessing, feature extraction  
- Configurable via YAML or argument options  
- Use of **CREMA-D** dataset (requires appropriate licensing / download)  
- Clear separation between feature types and fusion experiments  

---

## Repository Structure

```

.
├── .vscode/
├── config/
│   └── *.yaml         ← Configuration files for experiments
├── data/
│   └── (raw / processed data)
├── models/
│   └── (saved model weights)
├── runs/
│   └── (logs, tensorboard, outputs)
├── scripts/
│   └── data_preprocess.py
│   └── train.py
│   └── evaluate.py
│   └── extract_features.py
├── utils/
│   └── helpers.py
│   └── datasets.py
│   └── metrics.py
├── README.md
├── requirements.txt
└── .gitignore

````

- **config/**: YAML or JSON configs specifying hyperparameters  
- **data/**: holds raw or processed audio / features  
- **models/**: stores trained model files  
- **runs/**: outputs, logs, plots, evaluation results  
- **scripts/**: main entry-point scripts (preprocessing, training, evaluation)  
- **utils/**: helper modules for data loading, metrics, etc.

---

## Setup & Requirements

1. Clone the repository:

   ```bash
   git clone https://github.com/omidnaeej/Speech-Emotion-Recognition-Mel-Spectrograms-and-HuBERT-Embeddings.git
   cd Speech-Emotion-Recognition-Mel-Spectrograms-and-HuBERT-Embeddings
   ```

2. Create a Python virtual environment (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate      # on Linux / macOS
   venv\Scripts\activate         # on Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   The repository’s README currently lists:

   ```
   torch torchaudio transformers scikit-learn matplotlib pyyaml tqdm
   ```

   (These are minimal; your particular config files may require additional packages.)

---

## Usage

### Data Preparation

* Download and organize the **CREMA-D** dataset (or your dataset of choice) in `data/`.
* Use `scripts/data_preprocess.py` to:

  * Load raw audio files
  * Compute and save log-Mel spectrograms
  * Optionally extract and cache HuBERT embeddings (via a pretrained model)

Example:

```bash
python scripts/data_preprocess.py \
  --input_dir data/raw_audio \
  --output_dir data/processed \
  --config config/preprocess.yaml
```

### Training

* Train a model using either spectrograms, HuBERT embeddings, or fused features:

```bash
python scripts/train.py \
  --config config/train_mel.yaml
```

* For experiments combining features, adjust the config accordingly (e.g. feature fusion, model architecture).

### Evaluation & Inference

* Evaluate on a held-out test split:

```bash
python scripts/evaluate.py \
  --model_path models/best_model.pth \
  --config config/evaluate.yaml
```

* Optionally run inference on new audio files by adapting the evaluate or inference scripts.

### Visualization & Analysis

* After evaluation, confusion matrices, per-class metrics, and plots are generated and saved to `runs/...`
* You can inspect the logs, plot additional curves (e.g. ROC, precision-recall), etc.

---

## Configuration

Each YAML config (e.g. in `config/`) should typically contain:

* Dataset paths
* Feature type (e.g. `mel`, `hubert`, `fusion`)
* Model hyperparameters (learning rate, epochs, batch size, architecture)
* Loss and metric settings
* Paths for output, logging, saving models

By decoupling logic from configuration, you can easily run different experiments without changing code.

---

## Results & Metrics

* For each experiment you can log / compute:

  * Accuracy, precision, recall, F1-score (overall & per-class)
  * Confusion matrix
  * Loss curves over epochs
  * Comparative performance of spectrogram-only vs HuBERT-only vs fused

You can include a summary table in this README or link to a results file.

---

## Extending / Customizing

* **Add new datasets**: adapt `utils/datasets.py` and preprocessing to support new audio-emotion corpora
* **Change model architectures**: plug in transformers, CNNs, LSTMs, attention layers
* **Fusion techniques**: experiment with concatenation, attention-based fusion, gating
* **Feature extraction**: incorporate other embeddings (e.g. Wav2Vec, CPC, etc.)
* **Hyperparameter tuning**: wrap with grid search, Optuna, etc.
* **Deployment / inference**: wrap into a REST API or streaming inference service

---

## Citations & References

If you base your work on prior models or datasets, please cite:

* The **CREMA-D** dataset paper
* The HuBERT model paper
* Any additional architectures or techniques used

For example:

> * “HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units”
> * *CREMA-D: Crowd-sourced Emotional Mutimodal Actors Dataset*

---

## License

*(Specify a license here — e.g. MIT, Apache 2.0, etc.)*

---

## Acknowledgements

* Thanks to the creators of CREMA-D for making the dataset available
* Inspiration from existing SER research combining acoustic and self-supervised features
