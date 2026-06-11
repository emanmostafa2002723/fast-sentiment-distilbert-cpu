# Development Log - Fast Sentiment Analysis Using Distilled Transformers on CPU

## Week 1 - Dataset Exploration & DistilBERT Orientation

### Progress

Started by reading the DistilBERT paper (Sanh et al., 2019) and the original BERT paper (Devlin et al., 2019) to understand what distillation actually does to the architecture , 6 transformer layers instead of 12, no token-type embeddings, 40% fewer parameters. Also read through the HuggingFace fine-tuning example in the code references to understand the Trainer API before writing anything.

Decided to start with SST-2 because the sentences are short and training runs fast on CPU, which made it easier to iterate on the setup without waiting an hour per run. Loaded SST-2 via the GLUE benchmark using `load_dataset("glue", "sst2")`. Ran basic EDA: sentence length distribution, class balance, coverage at different `max_length` thresholds. SST-2 is roughly balanced (56% positive in training) and the sentences are short enough that `max_length=64` covers about 97% of them. Picked 64 as the tokenization cap for SST-2.

Set up the project folder structure, installed dependencies, and fixed all random seeds (Python `random`, NumPy, PyTorch, `PYTHONHASHSEED`) so runs are reproducible. This matters because we are comparing models and even a 0.5% accuracy difference is meaningful at the subset sizes we are using.

Built the first TF-IDF + Logistic Regression pipeline on a 3,000-sample balanced subset of SST-2 (1,500 pos + 1,500 neg). Used `ngram_range=(1,2)`, `max_features=50000`, and `sublinear_tf=True` based on standard text classification practice. Deliberately left stop words in negations like "not good" and "never boring" are discriminative for sentiment.

Did a first DistilBERT fine-tuning run on the same subset: 2 epochs, batch size 16, lr=2e-5, AdamW with weight decay 0.01, 10% linear warmup. Training took roughly 20–25 minutes on CPU.

### Key Decisions

Chose to use a balanced stratified subset for midterm experiments rather than the full training set. Full dataset training is out of scope until Week 3, but a 3,000-sample balanced subset is enough to see meaningful differences between the baseline and DistilBERT. Using a stratified split prevents accidentally giving one model a label imbalance advantage.

Kept SST-2 `max_length=64` after checking coverage. Going higher does not change accuracy meaningfully on these short sentences but makes tokenization and attention significantly slower on CPU relevant to Objective 3.

### Issues

HuggingFace's `Trainer` threw a `TypeError` when `evaluation_strategy` and `save_strategy` were set together with `load_best_model_at_end=True`. This was a version mismatch with the installed `transformers`. Fixed in the next run by either matching the parameter names (`eval_strategy` in newer versions) or dropping the conflicting args. Documented in commit so the fix is traceable.



**Commits this week:**
- `init: project structure, requirements, seed utilities`
- `feat: SST-2 loading and EDA, max_length coverage analysis`
- `feat: TF-IDF + LogReg baseline on SST-2 subset`
- `feat: first DistilBERT fine-tuning run on SST-2`
- `fix: Trainer argument compatibility for installed transformers version`



## Week 2 - Classical Baseline, DistilBERT Fine-Tuning, Efficiency Measurements

### Progress

Added IMDb as the second dataset (25,000 train / 25,000 test). IMDb reviews contain HTML artifacts `&amp;`, `<br />`, inline tags so we wrote a `clean_text` function that decodes HTML entities, strips tags, lowercases, and collapses whitespace. Negations and punctuation are kept intentionally (same reasoning as the SST-2 baseline).

Ran the same `max_length` coverage analysis on IMDb. Reviews are much longer than SST-2 sentences (median ~200 words vs. ~18). At `max_length=128`, a significant fraction gets truncated; at `max_length=256`, coverage is ~95%. At `max_length=512`, coverage is nearly complete but inference is roughly 4× slower than 256 with minimal accuracy gain on most reviews. Chose `IMDB_MAX_LEN=256`.

Built the classical baseline for both datasets. TF-IDF pipeline is the same configuration as Week 1. Measured training time, per-sample inference latency (100-sample benchmark with a 5-sample warmup), and serialized model size. The LR pipeline trains in under a second and infers in well under 1 ms per sample.

Fine-tuned DistilBERT on a 2,000-sample stratified IMDb subset (1,600 train / 400 val) and the 3,000-sample SST-2 subset. Ran 2 epochs on both. Used `load_best_model_at_end=True` and saved checkpoints per epoch to recover the best val-F1 checkpoint without overfitting on the small subset. Extracted per-step training loss and per-epoch validation loss/accuracy from `trainer.state.log_history` and plotted learning curves for the report.

Ran latency and memory benchmarks on the fine-tuned DistilBERT models: median inference time, 95th percentile, model parameter size in MB, and full process RSS. DistilBERT on CPU is significantly slower per sample than LR + TF-IDF, which is expected, this is one of the central trade-offs the project is investigating (Objective 3).

Consolidated everything into the final midterm notebook covering both datasets, with all outputs (figures, CSVs) saved to disk under `project4_midterm/`.

### Key Decisions

Used separate `max_length` values for the two datasets (64 for SST-2, 256 for IMDb) rather than a single shared value. A shared 256 would unnecessarily slow down SST-2 experiments; a shared 64 would truncate most IMDb reviews. Each dataset gets the value justified by its own EDA.

For the IMDb training split, we take a 2,000-sample stratified subset from the official 25,000-sample train set and use 20% of that subset as validation. The official test set is held out entirely — it will only be touched in the final evaluation. Using the official split boundary avoids any train/test contamination.

The latency benchmark excludes the first few samples as warmup to avoid measuring JIT compilation or cache cold-start effects. Results reflect steady-state CPU inference.

### Issues

`max_length=512` was the original default from DL_1. After running the coverage EDA, it was clear that 512 is wasteful for IMDb (adds a lot of padding at 95th percentile) and was adding about 30+ extra minutes to training for negligible accuracy change. Reduced to 256 after confirming coverage. All subsequent runs use 256 for IMDb.

The training history extraction from `trainer.state.log_history` had an off-by-one in mapping eval steps back to training steps when drawing the loss curve. Fixed by iterating over log entries by key rather than position.



**Commits this week:**
- `feat: IMDb loading, HTML cleaning, EDA, max_length justification`
- `feat: classical baseline for IMDb, latency + memory benchmarks`
- `fix: reduce IMDB_MAX_LEN 512→256 based on coverage EDA`
- `feat: DistilBERT fine-tuning on IMDb subset, 2 epochs`
- `feat: learning curve extraction from trainer log history`
- `fix: training history step alignment in loss plot`
- `feat: efficiency benchmarks, latency and memory comparison`
- `refactor: consolidate all midterm experiments into final notebook`

- ## Midterm Revision — Patch 1

### Changes made

- Added references [7] (Loshchilov & Hutter, AdamW) and [8] (Mikolov et al., word2vec) to reach ≥8 total references per grader feedback.
- Named specific GenAI tools in Section VII (ChatGPT GPT-4o, Claude, GitHub Copilot) to address grader comment on anonymous tool references.
- Added Dates and Responsible columns to Table IV (Timeline) with specific target dates per week.


## Week 3: Efficiency Analysis Notebook

### Progress

Built `07_efficiency_analysis.ipynb` from scratch this week.
Added latency benchmarking (median + p95, 100 samples, 5 warm-up),
memory profiling with psutil RSS, model size via torch.save,
Pareto frontier plot, and summary comparison table.
Results table seeded with FP32 baseline numbers from midterm.
INT8 and BiLSTM rows will be filled once those experiments complete.

### Commits this week
- feat: create efficiency analysis notebook
- feat: add imports to efficiency notebook
- feat: add latency measurement function
- feat: add memory profiling and model size functions
- feat: add Pareto frontier plot function
- feat: add results table with baseline numbers, generate Pareto plot
- feat: add summary table and CSV export

- week 4
- Esraa:
- notebook measures how efficient each model is on CPU, that is the deployment target.
All numbers come from the main notebook run (same seed, same models):
Model	Dataset	Acc %	Lat (ms)	Size (MB)
TF-IDF + LR	SST-2	81.65	0.92 ± 0.03	2.33
TF-IDF + LR	IMDb	89.50	1.34 ± 0.25	2.24
BiLSTM+GloVe+Attn	SST-2	85.32	2.61 ± 0.15	13.86
BiLSTM+GloVe+Attn	IMDb	88.44	7.71 ± 1.07	13.86
DistilBERT FP32	SST-2	90.71	49.94 ± 4.12	255.5
DistilBERT FP32	IMDb	91.07	99.25 ± 8.42	255.5
DistilBERT INT8	SST-2	89.45	17.83 ± 1.40	132.3
DistilBERT INT8	IMDb	89.91	49.89 ± 5.92	132.3
Setup
Accuracy before vs. after quantization
Latency benchmarks, mean, std, p95
Memory profiling (psutil RSS + tracemalloc)
Model size reduction
BiLSTM efficiency comparison
Pareto frontier: accuracy vs. latency
Report-ready overview figure
Summary table
