# 📊 Arabic Sentiment Analysis of Egyptian Dialect Tweets Using AraBERT

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-F9AB00?style=for-the-badge&logo=huggingface&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google_Colab-T4_GPU-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<p align="center">
  <b>Graduation Project — Bachelor of Data Science & Artificial Intelligence</b><br/>
  University College of Applied Sciences (UCAS), Gaza, Palestine
</p>

---

## 🧠 Overview

This repository contains the full codebase for our graduation project, which develops a **robust Arabic sentiment analysis system** for Egyptian dialect tweets by fine-tuning the **AraBERT** transformer model (`aubmindlab/bert-base-arabertv02`).

The project uses a **two-stage domain adaptation strategy**:

1. **Stage 1 — Domain Pretraining**: Fine-tune AraBERT on ArSarcasm-v2 Egyptian dialect data (full unfrozen layers)
2. **Stage 2 — Task Fine-tuning**: Continue training on the ASTD dataset with **Focal Loss** + **class weights** to handle class imbalance

This approach is compared against a simpler **Baseline** model trained only on raw ASTD tweets with 10/12 frozen layers.

---

## 🎯 Problem Statement

Egyptian dialect Arabic presents unique NLP challenges:
- Highly informal spelling and code-switching
- Significant class imbalance (NEG >> POS > NEUTRAL)
- Lack of large-scale labeled Egyptian dialect corpora

**Our solution:** A dialect-aware preprocessing pipeline + two-stage domain adaptation + Focal Loss to overcome these challenges.

---

## 🗄️ Datasets

| Dataset | Size | Purpose |
|---|---|---|
| [ASTD](https://github.com/mahmoudnabil/ASTD) | 3,315 tweets (after filtering OBJ class) | Primary training & evaluation |
| [ArSarcasm-v2](https://github.com/iabufarha/ArSarcasm-v2) | Egyptian dialect subset | Domain adaptation (Stage 1) |

**ASTD Class Distribution (after removing Objective class):**

| Class | Count | % |
|---|---|---|
| Negative | ~1,860 | ~56% |
| Positive | ~1,000 | ~30% |
| Neutral | ~455 | ~14% |

---

## ✨ Methodology

### Preprocessing Pipeline (10 Steps)
1. Remove `@mentions` and URLs
2. Remove emojis
3. Remove punctuation (Arabic + Latin)
4. Normalize repeated characters (`كتيييير` → `كتيير`)
5. Normalize Arabic letter variants (`أ إ آ` → `ا`, `ى` → `ي`, `ة` → `ه`)
6. Remove Arabic diacritics (Tashkeel: `َ ُ ِ ّ ْ`)
7. Remove extra whitespace
> ⚠️ **Stop-word removal and stemming were intentionally skipped** — these can destroy contextual meaning that AraBERT depends on.

### Model Architecture
```
aubmindlab/bert-base-arabertv02
├── 12 Bidirectional Transformer Encoder layers
├── WordPiece Tokenizer (max_length=128)
└── Classification Head → [NEG, NEUTRAL, POS]
```

### Training Setup

| | Baseline | Proposed |
|---|---|---|
| **Data** | ASTD raw text only | Stage 1: ArSarcasm-v2 Egyptian → Stage 2: ASTD clean |
| **Preprocessing** | None | Full 10-step pipeline |
| **Frozen layers** | 10/12 | 0 (full fine-tuning) |
| **Loss function** | Cross-Entropy | Focal Loss (γ=2.0) + class weights |
| **Epochs** | 3 | Stage 1: 3 + Stage 2: 7 |
| **Learning rate** | 2e-5 | Stage 1: 2e-5 → Stage 2: 1e-5 |
| **Optimizer** | AdamW | AdamW |

---

## 📊 Results
> ⚠️ Results below are from the final training run on Google Colab (T4 GPU).

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| **Baseline** | [64.16]% | [60.31]% | [56.44]% | [56.17]% |
| **Proposed** | [68.07]% | [68.67]% | [69.81]% | [67.82]% |
| **Improvement** | +[3.92]% | +[8.36]% | +[13.37]% | +[11.65]% |

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.12 | Core language |
| PyTorch | 2.1 | Deep learning framework |
| HuggingFace Transformers | latest | AraBERT model & Trainer |
| scikit-learn | latest | Metrics & evaluation |
| pandas | latest | Data handling |
| matplotlib / seaborn | latest | Visualization |
| arabert / farasapy | latest | Arabic NLP utilities |
| Google Colab T4 GPU | — | Training hardware |

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Randsalem19/arabic-sentiment-arabert.git
cd arabic-sentiment-arabert
```

### 2. Install Dependencies
```bash
pip install transformers datasets torch scikit-learn pandas matplotlib seaborn arabert farasapy
```

### 3. Prepare Datasets
You need two files:
- `Tweets.txt` — ASTD dataset (tab-separated: `tweet\tlabel`)
- `ArSarcasm-v2_train.csv` and `ArSarcasm-v2_test.csv` — from the ArSarcasm-v2 repository

### 4. Run the Notebook
Open `GP_Arabic_Sentiment_Analysis.ipynb` in Google Colab and run all cells.  
When prompted, upload the dataset files using the Colab file uploader.

---

## 📁 Repository Structure

```
├── GP_Arabic_Sentiment_Analysis.ipynb   # Main training & evaluation notebook
├── README.md                            # This file
└── results/                             # Generated after running (classification reports, CSVs)
```

---

## 👥 Authors

| Name | Role |
|---|---|
| **Rand Majed Salem** | [GitHub](https://github.com/Randsalem19) · [LinkedIn](https://linkedin.com/in/rand-majed-salem) · [Kaggle](https://kaggle.com/randsalem) |
| Nagham E. Zidiah | Team Member |
| Raghad M. ALSerhy | Team Member |
| Sara A. Alamour | Team Member |
| **Dr. Esraa Ferwana** | Academic Advisor |

---



<p align="center">Made with ❤️ at UCAS, Gaza, Palestine 🇵🇸</p>
