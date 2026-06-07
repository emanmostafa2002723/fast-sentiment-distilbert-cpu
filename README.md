
# Sentiment Analysis on CPU

**DistilBERT vs TF-IDF — efficiency-first comparison**

> Deep Learning Midterm Project — Queen's University
> **Team:** Arwa Elgazar · Eman Elsayed · Esraa Nematalla

---

## What This Is

We compared three approaches to sentiment analysis under a CPU-only constraint — no GPUs allowed:

- TF-IDF + Logistic Regression (baseline)
- DistilBERT (fine-tuned)
- INT8 Quantized DistilBERT (optimized)

Tested on two datasets: **SST-2** (short movie snippets) and **IMDb** (long reviews). The core question: *what's the best accuracy-vs-speed tradeoff when you can't use a GPU?*

---

## Results

| Dataset | Model | Accuracy | Speed |
|---------|-------|----------|-------|
| SST-2 | TF-IDF + LR | 72% | Very fast |
| SST-2 | DistilBERT | 86% | Slow |
| IMDb | TF-IDF + LR | 82.7% | Fast |
| IMDb | DistilBERT | 89% | Very slow |

INT8 quantization brought DistilBERT's latency and memory down significantly with almost no accuracy loss — making it the most practical option for real deployment.

---

## Files

```
sentiment-analysis-cpu/
├── sentiment_analysis_midterm_Notebook_team_Eman_Esraa_Arwa.ipynb
├── ANALYSIS_BASED_sentiment_94.ipynb
└── README.md
```

**Main notebook** — data loading, preprocessing, TF-IDF baseline, DistilBERT fine-tuning, and initial evaluation.

**Analysis notebook** — INT8 quantization, detailed latency benchmarking, memory tracking, error analysis, and the 94% accuracy run.

---

## Setup

```bash
pip install transformers datasets scikit-learn torch evaluate matplotlib seaborn
```

Requires Python 3.9+, PyTorch (CPU build), HuggingFace Transformers, and scikit-learn.

---

## Approach

**Baseline:** TF-IDF with bigrams (`max_features=50000`) + Logistic Regression. Extremely fast, no context awareness.

**DistilBERT:** `distilbert-base-uncased` fine-tuned end-to-end via the HuggingFace Trainer API. Best accuracy, but slow on CPU.

**Quantized DistilBERT:** Applied INT8 dynamic quantization to all linear layers. Near-identical accuracy, noticeably faster — best option for production.

All experiments use `SEED = 42` for reproducibility.

---

## Team

| Name | Work |
|------|------|
| Arwa Elgazar | Data preprocessing, EDA, TF-IDF baseline |
| Eman Elsayed | DistilBERT training, tokenization, model tuning |
| Esraa Nematalla | Quantization, efficiency analysis, evaluation |

---

## Takeaway

Raw transformers are too slow for CPU inference without optimization. Quantization closes most of that gap — it's what makes DistilBERT actually deployable in a real system.

