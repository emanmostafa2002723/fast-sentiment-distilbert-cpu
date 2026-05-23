Nice — now that I’ve actually looked inside both notebooks, we can upgrade your repo from “good student submission” → “this looks like a serious ML project”.

Here’s an **updated README that reflects the real notebook content (including quantization + actual workflow)**:

---

# 🚀 Fast Sentiment Analysis on CPU

### DistilBERT vs TF-IDF with Efficiency Optimization

> 🎓 Deep Learning Midterm Project — Queen’s University
> 👥 **Team:** Arwa Elgazar · Eman Elsayed · Esraa Nematalla

---

## 📌 Overview

This project investigates **efficient sentiment analysis under strict CPU-only constraints** by comparing:

* 📊 **TF-IDF + Logistic Regression (baseline)**
* 🤖 **DistilBERT (fine-tuned)**
* ⚡ **INT8 Quantized DistilBERT (optimized)**

We evaluate models across:

* 🧾 **SST-2 (short text)**
* 📚 **IMDb (long reviews)**

🎯 Goal: **Find the best trade-off between accuracy, latency, and model size**

---

## 🧠 Key Contributions

* ✅ Full **CPU-only pipeline** (training + inference)
* ✅ Direct comparison: **classical ML vs transformers**
* ✅ **Latency benchmarking** (real inference cost)
* ✅ **INT8 dynamic quantization** for efficiency
* ✅ Token-length optimization per dataset
* ✅ Reproducible experiments (fixed seeds)

---

## 📊 Results Summary

| Dataset | Model       | Accuracy | Latency      | Insight              |
| ------- | ----------- | -------- | ------------ | -------------------- |
| SST-2   | TF-IDF + LR | 72%      | ⚡ Very fast  | Lightweight baseline |
| SST-2   | DistilBERT  | 86%      | 🐢 Slow      | Best performance     |
| IMDb    | TF-IDF + LR | 82.7%    | ⚡ Fast       | Strong baseline      |
| IMDb    | DistilBERT  | 89%      | 🐢 Very slow | Highest accuracy     |

### ⚡ Optimization Insight

* INT8 Quantization significantly reduces **latency & memory**
* Minimal impact on accuracy → **best deployment candidate**

---

## 🗂️ Repository Structure

```bash
📦 sentiment-analysis-cpu
├── 📓 sentiment_analysis_midterm_Notebook_team_Eman_Esraa_Arwa.ipynb
├── 📓 ANALYSIS_BASED_sentiment_94.ipynb
└── 📄 README.md
```

---

## 📓 Notebooks Breakdown

### 🔹 1. Main Midterm Notebook

`sentiment_analysis_midterm_Notebook_team_Eman_Esraa_Arwa.ipynb`

Covers the **core pipeline (Week 1–2):**

* 📊 Data loading (SST-2 + IMDb)
* 🧹 Text preprocessing (HTML removal, cleaning)
* 🔡 TF-IDF feature engineering
* 📈 Logistic Regression baseline
* 🤖 DistilBERT fine-tuning (HuggingFace Trainer)
* 📏 Evaluation (Accuracy, F1-score)
* ⏱️ Initial latency measurements

---

### 🔹 2. Advanced Analysis Notebook

`ANALYSIS_BASED_sentiment_94.ipynb`

Extends the project with **optimization & deeper analysis:**

* ⚡ **INT8 Dynamic Quantization**
* ⏱️ Detailed latency benchmarking
* 📉 Memory usage tracking
* 📈 Performance comparison across models
* 🔍 Error analysis
* 🧪 Experimental improvements (94% accuracy run)

---

## ⚙️ Setup

```bash
pip install transformers datasets scikit-learn torch evaluate matplotlib seaborn
```

### 🧩 Requirements

* Python 3.9+
* PyTorch (CPU)
* HuggingFace Transformers
* scikit-learn

---

## 🧪 Methodology

### 📚 Datasets

* SST-2 (GLUE benchmark)
* IMDb reviews
* Stratified sampling applied

---

### 📊 Baseline Model

```python
TfidfVectorizer(ngram_range=(1,2), max_features=50000)
+ LogisticRegression
```

✔ Extremely fast
❌ Limited contextual understanding

---

### 🤖 DistilBERT

* Model: `distilbert-base-uncased`
* Fine-tuned end-to-end
* Trainer API (HuggingFace)

✔ Strong semantic understanding
❌ High latency on CPU

---

### ⚡ Quantized DistilBERT

```python
torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

✔ Reduced model size
✔ Faster inference
✔ Minimal accuracy drop

👉 **Best balance for real-world deployment**

---

## ⏱️ Performance Trade-off

| Model          | Speed       | Accuracy | Use Case                      |
| -------------- | ----------- | -------- | ----------------------------- |
| TF-IDF + LR    | 🔥 Fastest  | Medium   | Real-time systems             |
| DistilBERT     | 🐢 Slow     | High     | Offline / high-accuracy tasks |
| Quantized BERT | ⚖️ Balanced | High     | Production (CPU)              |

---

## 🔁 Reproducibility

* Fixed seed: `SEED = 42`
* Applied across:

  * Python
  * NumPy
  * PyTorch

---

## 🔮 Future Work

* 📏 Sequence length ablation (64–512)
* 🧊 Layer freezing experiments
* 📉 Training size scaling
* 🔄 Cross-dataset generalization
* 🚀 Deploy as API (Flask / FastAPI)

---

## 👥 Team Contributions

| Name                | Contribution                                          |
| ------------------- | ----------------------------------------------------- |
| **Arwa Elgazar**    | Data preprocessing, EDA, TF-IDF baseline, experiments |
| **Eman Elsayed**    | DistilBERT training, tokenization, model tuning       |
| **Esraa Nematalla** | Efficiency analysis, quantization, evaluation         |

---

## ⭐ Final Takeaway

> On CPU, raw transformers are **not practical without optimization**
> → Quantization makes them **actually usable in real systems**

---

If you want next step (highly recommended for your CV 👇):

* Add **badges (Python, HuggingFace, License)**
* Add **plots (latency vs accuracy curve)**
* Add **demo section (even screenshots)**
* Rename repo to something stronger like:
  👉 `cpu-efficient-sentiment-analysis`

I can also turn this into a **top 1% GitHub portfolio repo** if you want.
