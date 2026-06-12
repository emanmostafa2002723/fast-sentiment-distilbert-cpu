# Development Log - Fast Sentiment Analysis Using Distilled Transformers on CPU

## Week 1 - Dataset Exploration & DistilBERT Orientation

### Progress

Started by reading the DistilBERT paper (Sanh et al., 2019) and the original BERT paper
(Devlin et al., 2019) to understand what distillation actually does to the architecture ,
6 transformer layers instead of 12, no token-type embeddings, 40% fewer parameters. Also
read through the HuggingFace fine-tuning example in the code references to understand the
Trainer API before writing anything.

Decided to start with SST-2 because the sentences are short and training runs fast on CPU,
which made it easier to iterate on the setup without waiting an hour per run. Loaded SST-2
via the GLUE benchmark using `load_dataset("glue", "sst2")`. Ran basic EDA: sentence length
distribution, class balance, coverage at different `max_length` thresholds. SST-2 is roughly
balanced (56% positive in training) and the sentences are short enough that `max_length=64`
covers about 97% of them. Picked 64 as the tokenization cap for SST-2.

Set up the project folder structure, installed dependencies, and fixed all random seeds
(Python `random`, NumPy, PyTorch, `PYTHONHASHSEED`) so runs are reproducible. This matters
because we are comparing models and even a 0.5% accuracy difference is meaningful at the
subset sizes we are using.

Built the first TF-IDF + Logistic Regression pipeline on a 3,000-sample balanced subset of
SST-2 (1,500 pos + 1,500 neg). Used `ngram_range=(1,2)`, `max_features=50000`, and
`sublinear_tf=True` based on standard text classification practice. Deliberately left stop
words in negations like "not good" and "never boring" are discriminative for sentiment.

Did a first DistilBERT fine-tuning run on the same subset: 2 epochs, batch size 16, lr=2e-5,
AdamW with weight decay 0.01, 10% linear warmup. Training took roughly 20–25 minutes on CPU.

### Key Decisions

Chose to use a balanced stratified subset for midterm experiments rather than the full
training set. Full dataset training is out of scope until Week 3, but a 3,000-sample balanced
subset is enough to see meaningful differences between the baseline and DistilBERT. Using a
stratified split prevents accidentally giving one model a label imbalance advantage.

Kept SST-2 `max_length=64` after checking coverage. Going higher does not change accuracy
meaningfully on these short sentences but makes tokenization and attention significantly
slower on CPU relevant to Objective 3.

### Issues

HuggingFace's `Trainer` threw a `TypeError` when `evaluation_strategy` and `save_strategy`
were set together with `load_best_model_at_end=True`. This was a version mismatch with the
installed `transformers`. Fixed in the next run by either matching the parameter names
(`eval_strategy` in newer versions) or dropping the conflicting args. Documented in commit
so the fix is traceable.

**Commits this week:**
- `init: project structure, requirements, seed utilities`
- `feat: SST-2 loading and EDA, max_length coverage analysis`
- `feat: TF-IDF + LogReg baseline on SST-2 subset`
- `feat: first DistilBERT fine-tuning run on SST-2`
- `fix: Trainer argument compatibility for installed transformers version`

---

## Week 2 - Classical Baseline, DistilBERT Fine-Tuning, Efficiency Measurements

### Progress

Added IMDb as the second dataset (25,000 train / 25,000 test). IMDb reviews contain HTML
artifacts `&amp;`, `<br />`, inline tags so we wrote a `clean_text` function that decodes
HTML entities, strips tags, lowercases, and collapses whitespace. Negations and punctuation
are kept intentionally (same reasoning as the SST-2 baseline).

Ran the same `max_length` coverage analysis on IMDb. Reviews are much longer than SST-2
sentences (median ~200 words vs. ~18). At `max_length=128`, a significant fraction gets
truncated; at `max_length=256`, coverage is ~95%. At `max_length=512`, coverage is nearly
complete but inference is roughly 4× slower than 256 with minimal accuracy gain on most
reviews. Chose `IMDB_MAX_LEN=256`.

Built the classical baseline for both datasets. TF-IDF pipeline is the same configuration
as Week 1. Measured training time, per-sample inference latency (100-sample benchmark with
a 5-sample warmup), and serialized model size. The LR pipeline trains in under a second and
infers in well under 1 ms per sample.

Fine-tuned DistilBERT on a 2,000-sample stratified IMDb subset (1,600 train / 400 val) and
the 3,000-sample SST-2 subset. Ran 2 epochs on both. Used `load_best_model_at_end=True` and
saved checkpoints per epoch to recover the best val-F1 checkpoint without overfitting on the
small subset. Extracted per-step training loss and per-epoch validation loss/accuracy from
`trainer.state.log_history` and plotted learning curves for the report.

Ran latency and memory benchmarks on the fine-tuned DistilBERT models: median inference
time, 95th percentile, model parameter size in MB, and full process RSS. DistilBERT on CPU
is significantly slower per sample than LR + TF-IDF, which is expected, this is one of the
central trade-offs the project is investigating (Objective 3).

Consolidated everything into the final midterm notebook covering both datasets, with all
outputs (figures, CSVs) saved to disk under `project4_midterm/`.

### Key Decisions

Used separate `max_length` values for the two datasets (64 for SST-2, 256 for IMDb) rather
than a single shared value. A shared 256 would unnecessarily slow down SST-2 experiments;
a shared 64 would truncate most IMDb reviews. Each dataset gets the value justified by its
own EDA.

For the IMDb training split, we take a 2,000-sample stratified subset from the official
25,000-sample train set and use 20% of that subset as validation. The official test set is
held out entirely — it will only be touched in the final evaluation. Using the official split
boundary avoids any train/test contamination.

The latency benchmark excludes the first few samples as warmup to avoid measuring JIT
compilation or cache cold-start effects. Results reflect steady-state CPU inference.

### Issues

`max_length=512` was the original default from DL_1. After running the coverage EDA, it was
clear that 512 is wasteful for IMDb (adds a lot of padding at 95th percentile) and was
adding about 30+ extra minutes to training for negligible accuracy change. Reduced to 256
after confirming coverage. All subsequent runs use 256 for IMDb.

The training history extraction from `trainer.state.log_history` had an off-by-one in
mapping eval steps back to training steps when drawing the loss curve. Fixed by iterating
over log entries by key rather than position.

**Commits this week:**
- `feat: IMDb loading, HTML cleaning, EDA, max_length justification`
- `feat: classical baseline for IMDb, latency + memory benchmarks`
- `fix: reduce IMDB_MAX_LEN 512→256 based on coverage EDA`
- `feat: DistilBERT fine-tuning on IMDb subset, 2 epochs`
- `feat: learning curve extraction from trainer log history`
- `fix: training history step alignment in loss plot`
- `feat: efficiency benchmarks, latency and memory comparison`
- `refactor: consolidate all midterm experiments into final notebook`

---

## Midterm Revision — Patch 1

### Changes made

- Added references [7] (Loshchilov & Hutter, AdamW) and [8] (Mikolov et al., word2vec)
  to reach ≥8 total references per grader feedback.
- Named specific GenAI tools in Section VII (ChatGPT GPT-4o, Claude, GitHub Copilot) to
  address grader comment on anonymous tool references.
- Added Dates and Responsible columns to Table IV (Timeline) with specific target dates
  per week.

---

## Week 3 - Full-Data Scaling, BiLSTM Baseline, INT8 Quantisation, Ablation Studies

### Progress

**Efficiency analysis notebook.** Started the week by building `07_efficiency_analysis.ipynb`
as a dedicated benchmarking module separate from the training notebooks. The notebook
consolidates latency measurement (median and p95, 100 samples, 5 warm-up), memory profiling
via `psutil` RSS, model size measurement via `torch.save`, Pareto frontier plotting, and a
unified summary comparison table. At this stage the results table was seeded with the FP32
baseline numbers carried over from the midterm; the INT8 and BiLSTM rows were left as
placeholders to be filled once those experiments completed later in the week. The Pareto
plot function was written generically so new model configurations can be added by appending
a single row to the results dictionary without changing any plotting code.

**Scaling up training data.** The midterm experiments used deliberately small stratified
subsets — 3,000 SST-2 samples and 2,000 IMDb samples — to keep iteration time short on CPU.
Beginning this week, we expanded the training data for all model families. TF-IDF + LR was
retrained on the full SST-2 training split (67,349 samples) and the full IMDb training split
(25,000 samples); training time stays under 30 seconds for both so there is no practical
constraint. DistilBERT is more tightly constrained by CPU training time. Full SST-2
fine-tuning would take roughly four to five hours and full IMDb at `max_length=256` would
approach six hours per two-epoch run. We therefore expanded to a 6,000-sample SST-2 subset
and a 4,000-sample IMDb subset — approximately double the midterm sizes — keeping each
two-epoch run under 30 minutes for SST-2 and under 35 minutes for IMDb. The training-size
ablation described below sweeps the full practical range and confirms that these expanded
subsets capture most of the available accuracy gain before the curve flattens.

**BiLSTM baseline.** Added a Bidirectional LSTM classifier as the required second deep
baseline, addressing the TA feedback that the project needed at least two model families
beyond the classical pipeline. The architecture uses a trainable embedding layer initialised
with GloVe 6B 300d pre-trained vectors; tokens absent from the GloVe vocabulary receive
zero-mean random uniform initialisation in `[-0.01, 0.01]`. A single BiLSTM layer with 128
hidden units per direction (256 combined) processes the embedded sequence; dropout of 0.3
is applied to the BiLSTM output; a linear head maps the concatenated final forward and
backward hidden states to two logits. Text is tokenised with a whitespace-and-punctuation
tokeniser, the vocabulary is capped at 30,000 tokens by frequency, and sequences are padded
or truncated to `max_length=64` for SST-2 and `max_length=256` for IMDb — the same cutoffs
used for DistilBERT, so sequence length is not a confounding variable in the three-way
comparison. Training uses Adam (lr=1e-3), batch size 32, gradient clipping at 1.0,
cross-entropy loss, 3 epochs per dataset. The 100-sample latency benchmark with 5 warm-up
samples from the midterm was applied using the same `time.perf_counter()` loop, allowing
direct comparison across all three model families.

On the 6K SST-2 subset, the BiLSTM reaches 79.5% validation accuracy (macro-F1 0.795)
after 3 epochs. On the 4K IMDb subset, it reaches 85.5% validation accuracy (macro-F1
0.854). Both results sit between TF-IDF and DistilBERT, which is the expected ordering:
the BiLSTM has learned sequential representations but lacks large-scale pre-trained
contextual knowledge. Median per-sample inference latency is 3.7 ms on SST-2 and 9.2 ms
on IMDb, substantially faster than DistilBERT but much slower than TF-IDF. Serialised model
size is approximately 38 MB, dominated by the 300-dimensional embedding matrix.

The BiLSTM results were used to fill the previously empty rows in the efficiency analysis
notebook's summary table and Pareto plot.

**INT8 dynamic quantisation.** Applied `torch.quantization.quantize_dynamic()` to both
fine-tuned DistilBERT checkpoints, targeting `{torch.nn.Linear}` as the quantised layer
set. Dynamic quantisation stores linear weights as INT8 and dequantises them on the fly
during matrix multiplications at inference time; no calibration dataset is required.
Serialised checkpoint size dropped from 267.9 MB to 66.8 MB — a 4.0× reduction matching
the midterm prediction. Median inference latency on SST-2 dropped from 18.5 ms to 12.0 ms
(35.1% reduction) and on IMDb from 51.4 ms to 33.3 ms (35.2% reduction), both within the
predicted 30–40% range. Validation accuracy on SST-2 dropped 0.5 pp (86.0% → 85.5%) and
on IMDb 0.25 pp (89.0% → 88.75%), both well within the <1 pp threshold. The static
quantisation fallback was not needed. INT8 results were added to the efficiency analysis
notebook, completing the summary table and updating the Pareto frontier plot.

**Sequence-length ablation on IMDb.** Swept `max_length ∈ {64, 128, 256, 512}` on the 4K
IMDb subset using the INT8-quantised DistilBERT, since that reflects the intended deployment
model. Validation accuracy rises from 84.1% at 64 to 87.8% at 128, 88.75% at 256, and
89.0% at 512. Median inference latency follows the expected growth pattern from
self-attention: 9.1 ms at 64, 16.3 ms at 128, 33.3 ms at 256, 67.8 ms at 512. The
coverage-latency-accuracy curve confirms `max_length=256` as the near-optimal operating
point — the additional 0.25 pp gain from 256 to 512 does not justify a 2× latency increase
for CPU deployment.

**Training-set size ablation.** Swept training subset sizes {500, 1K, 2K, 4K, 8K} for
DistilBERT and {500, 1K, 2K, 4K, full 25K} for TF-IDF on IMDb, holding evaluation on the
fixed 400-sample validation set from the midterm. TF-IDF accuracy rises from 74.1% at 500
samples to 84.9% at 4K and 88.2% at the full 25K split. DistilBERT shows steeper early
gains — 80.3% at 500 samples, 87.0% at 1K — but the accuracy advantage of DistilBERT over
TF-IDF shrinks from 6.2 pp at 2K (matching the midterm result exactly) to 2.2 pp at 4K,
confirming the midterm hypothesis that the transformer's advantage diminishes as the
classical model accumulates more lexical evidence from longer reviews.

**Memory profiling.** Completed the `psutil` RSS measurements planned in the midterm,
integrating them into the efficiency analysis notebook at three checkpoints: before model
loading (Python baseline RSS), immediately after the model and tokenizer are loaded into
RAM, and after completing the 100-sample benchmark. DistilBERT FP32 adds approximately
900 MB above the Python baseline; INT8 reduces this to approximately 670 MB. TF-IDF + LR
adds approximately 180 MB (vocabulary and weight matrices are larger in RAM than the
pickled size suggests). The BiLSTM adds approximately 420 MB, dominated by the GloVe
embedding matrix held in memory.

### Key Decisions

Chose the BiLSTM rather than BERT-base without distillation as the second deep baseline.
BERT-base fine-tuning at `max_length=256` on CPU would take several hours per two-epoch
run, making any ablation sweep over it impractical within the project timeline. The BiLSTM
satisfies the rubric requirement for a second deep-learning baseline while remaining
tractable, and it occupies the natural efficiency middle ground between TF-IDF and
DistilBERT — which makes it the most informative third point on the Pareto frontier.

GloVe 6B 300d was chosen over random embedding initialisation because the training subsets
(4K–6K samples) are too small to learn useful dense representations from scratch. Pre-trained
embeddings give the BiLSTM access to lexical knowledge without contextual pre-training,
making the BiLSTM vs. DistilBERT comparison specifically about the contribution of
contextual pre-training rather than embedding quality.

The sequence-length ablation was run on the INT8 model rather than FP32 so that the reported
latency numbers directly reflect the deployment configuration. Running the ablation only on
FP32 would require a second sweep to verify that the optimal operating point is the same
under INT8 — which is unnecessary given that quantisation affects all sequence lengths
proportionally.

### Issues

The `torchtext` GloVe download API raised a `RuntimeError` about a missing
`build_vocab_from_iterator` utility in the installed library version. Fixed by downloading
`glove.6B.zip` directly from the Stanford NLP distribution page and writing a standalone
`load_glove_vectors(path, vocab, dim=300)` function that parses the raw text file and
constructs the embedding matrix. The resulting tensor is cached to disk as a `.pt` file so
subsequent runs skip the parse step entirely.

The INT8 quantised DistilBERT raised a `RuntimeError` on the first inference call with a
device placement message that was initially misleading. The actual cause was a dtype mismatch:
`DataCollatorWithPadding` was returning `input_ids` as `torch.int32` in one code path, and
the quantised model's embedding lookup requires `torch.long` (int64). The FP32 model
silently promotes int32 internally; the quantised model does not. Fixed by adding an explicit
`.long()` cast to `input_ids` inside the inference wrapper and verified against single-sample
spot checks before running the full benchmark.

The training-size ablation at the 8K point on IMDb took approximately 43 minutes per
two-epoch run, slightly over the 35-minute informal budget. All five DistilBERT checkpoints
from the ablation sweep were saved to disk. The `RUN_TRAINING=False` checkpoint-cache flag
introduced this week prevents completed ablation points from being re-run if the kernel is
restarted.

**Commits this week:**
- `feat: create efficiency analysis notebook`
- `feat: add imports to efficiency notebook`
- `feat: add latency measurement function`
- `feat: add memory profiling and model size functions`
- `feat: add Pareto frontier plot function`
- `feat: add results table with baseline numbers, generate Pareto plot`
- `feat: add summary table and CSV export`
- `feat: expand DistilBERT fine-tuning to larger subsets (6K SST-2, 4K IMDb)`
- `feat: BiLSTM baseline — vocab builder, GloVe loader, model definition`
- `feat: BiLSTM training loop and evaluation on SST-2 and IMDb expanded subsets`
- `feat: INT8 dynamic quantisation of both DistilBERT checkpoints`
- `feat: fill INT8 and BiLSTM rows in efficiency notebook summary table`
- `feat: sequence-length ablation sweep on IMDb — 64 / 128 / 256 / 512`
- `feat: training-set size ablation (500 to 8K) for DistilBERT and TF-IDF on IMDb`
- `feat: memory profiling with psutil RSS at load and inference checkpoints`
- `fix: GloVe loader — replace deprecated torchtext API with direct file parser`
- `fix: INT8 inference dtype mismatch — cast input_ids to torch.long before forward pass`

---

## Week 4 - Layer-Freezing Curve, Cross-Dataset Evaluation, Error Analysis, Final Report

### Progress

**Layer-freezing efficiency curve.** This is the project's original contribution described
in the midterm planned-work section. Trained five DistilBERT variants by progressively
freezing 0, 1, 2, 3, and 4 of the six transformer layers from the bottom of the network,
leaving the upper layers and the classification head trainable. Freezing is implemented by
iterating over `model.distilbert.transformer.layer[i]` for each target index `i` and calling
`param.requires_grad = False` on all parameters in that block before entering the training
loop. All five variants are trained under identical hyperparameters on the 6K SST-2 and 4K
IMDb subsets with the same random seed.

Validation F1 on SST-2 across the five configurations: 0 frozen → 86.8%; 1 frozen → 86.5%;
2 frozen → 86.2%; 3 frozen → 85.1%; 4 frozen → 83.4%. Training time decreases monotonically
from approximately 15 minutes at 0 frozen layers to approximately 12 minutes at 4 frozen
layers on SST-2 — a 20% reduction. On IMDb, training time drops from approximately 33
minutes to approximately 26 minutes (21% reduction). The accuracy cost at 2 frozen layers
is 0.6 pp on SST-2 and 0.7 pp on IMDb, slightly above the <0.5 pp midterm hypothesis but
within a practically acceptable range for training-efficiency-prioritised settings.

The hypothesised inference latency reduction from JIT-optimised frozen subgraphs did not
materialise. We tested TorchScript tracing (`torch.jit.trace`) on each frozen-layer variant
and observed a 2–4% median latency improvement at most, which is within run-to-run
measurement noise and was not consistent across repeated benchmark windows. This is reported
as an explicit null result: layer-freezing is a training-efficiency technique in this
experimental setting, not an inference-efficiency one. Inference latency is effectively
unchanged between frozen and unfrozen variants regardless of how many layers are frozen.
Quantisation remains the only robustly demonstrated lever for inference-time compression
in this project, and the Pareto frontier uses inference latency on the x-axis accordingly,
on which frozen and unfrozen variants are equivalent.

**Pareto frontier.** Produced the final Pareto frontier plot in the efficiency analysis
notebook covering all model configurations benchmarked throughout the project: TF-IDF + LR
(full training data), BiLSTM (expanded subset), DistilBERT FP32 (midterm subset),
DistilBERT FP32 (expanded subset), and DistilBERT INT8 (expanded subset). On the
inference-efficiency frontier, TF-IDF dominates at low latency and DistilBERT INT8 dominates
at high accuracy; the BiLSTM sits off the frontier — neither the fastest at its accuracy
level nor the most accurate at its latency level. A separate training-efficiency plot presents
DistilBERT frozen-layer variants on a training-time versus validation-F1 frontier, where
the 2-layer-frozen INT8 variant is Pareto-dominant among all transformer configurations on
both datasets.

**Cross-dataset evaluation.** Evaluated the SST-2-fine-tuned DistilBERT checkpoint (6K
subset, 0 frozen layers) on the full IMDb test set and the IMDb-fine-tuned model on the full
SST-2 validation set (872 samples). SST-2 → IMDb transfer: 78.3% accuracy, a 12.9 pp drop
compared to the IMDb in-domain result. IMDb → SST-2 transfer: 82.1% accuracy, a 5.0 pp drop
compared to the SST-2 in-domain result. The asymmetry is consistent with the expectation
that models trained on long, contextually rich IMDb reviews develop more transferable
representations than models trained on short SST-2 phrase fragments. Both results confirm
that in-domain fine-tuning is important even between two datasets that are both binary
English movie sentiment tasks.

**Error analysis.** Collected all misclassified validation examples from TF-IDF, BiLSTM,
and DistilBERT on both datasets and drew a stratified 200-sample set from each
model-dataset combination. Each sample was categorised into one of four bins: (a) negation
— examples where a negation marker is the likely source of the error; (b) sarcasm and irony
— examples where surface polarity is reversed by pragmatic context; (c) domain-specific
vocabulary — technical terms, named entities, or slang outside the training distribution;
(d) ambiguous or borderline sentiment — examples where a human annotator would plausibly
disagree on the label. On SST-2, negation is the dominant error category for TF-IDF (34% of
its errors), confirming the limitation of bigram features for long-range negation. DistilBERT's
primary error category on SST-2 is sarcasm and irony (23% of its errors), where contextual
representations alone are insufficient without pragmatic inference. The BiLSTM shows
intermediate negation error rates (19%), better than TF-IDF but worse than DistilBERT,
consistent with its sequential modeling capability.

**Full held-out evaluation.** Evaluated the final fine-tuned and quantised models on the
official IMDb test set (25,000 samples) and SST-2 validation set (872 samples). Sequential
single-sample inference over the 25K IMDb reviews with DistilBERT FP32 at `max_length=256`
took approximately 23 minutes; refactoring to a batched inference loop at batch size 64
reduced this to approximately 9 minutes without changing any numerical results. Final
held-out accuracy: DistilBERT INT8 achieves 91.2% on IMDb and 87.1% on SST-2; TF-IDF
trained on the full training splits achieves 89.3% on IMDb and 83.6% on SST-2; BiLSTM
achieves 87.8% on IMDb and 81.4% on SST-2. All numbers are higher than the midterm
validation results, consistent with the larger training sets used in Weeks 3 and 4.

**Final report.** Addressed all remaining TA feedback items not covered in the Midterm
Revision Patch. The reference list was expanded to 9 entries: Hochreiter and Schmidhuber
(1997) added for the LSTM architecture, Pennington et al. (2014) added for GloVe embeddings,
and Jacob et al. (2018) added for post-training integer quantisation. Generative AI tool
usage is now named explicitly throughout: ChatGPT-4o was used for concept clarification
(DistilBERT distillation mechanics) and debugging (Trainer API version mismatch, INT8 dtype
error); GitHub Copilot was used for boilerplate suggestions in the benchmarking utilities;
all experimental results, figures, tables, and analysis are original. Formal metric
definitions were added to the methodology section with mathematical equations for accuracy,
precision, recall, and macro-F1. The project timeline table was updated to include a
responsible-student column and specific target dates within each week. BiLSTM results were
added to all comparison tables and figures and discussed throughout the final report.

### Key Decisions

Reporting the layer-freezing inference latency result as an explicit null finding rather
than omitting it was deliberate. The midterm hypothesised a 15–25% latency reduction from
JIT optimisation of frozen subgraphs; the observed result was 2–4% with no consistency
across runs. Honest null results are scientifically important — they clarify the actual
mechanism at work (reduced gradient computation during training, not reduced activation
computation during inference) and prevent the reader from misinterpreting training-time
savings as inference-time savings.

For the full held-out evaluation, the larger-subset checkpoint (6K SST-2, 4K IMDb) with
INT8 quantisation was used rather than the midterm checkpoint (3K/2K). The training-size
ablation confirms that the larger subsets recover meaningfully more accuracy, so using the
midterm checkpoint for the final evaluation would understate the model's realistic
performance.

### Issues

The layer-freezing sweep required ten training runs (five layer configurations × two
datasets). IMDb runs took approximately 30–35 minutes each and SST-2 runs took approximately
12–15 minutes each, totalling approximately seven hours of CPU time distributed across
multiple sessions. The `RUN_TRAINING` checkpoint-cache flag from Week 3 was essential —
without it, a kernel restart would have invalidated all completed runs.

TorchScript tracing for the JIT latency experiment raised a `TracerWarning` about tensor
shapes changing between calls, caused by dynamic per-batch padding producing variable-length
input tensors. Switched to `torch.jit.script` as an alternative, but the latency gain
remained in the 2–4% range and inconsistent across measurement windows. Both tracing and
scripting results are retained in the notebook with raw latency distributions so the null
result is fully reproducible.

Refactoring the evaluation function to accept a `DataLoader` for batched inference required
changes to how `DataCollatorWithPadding` was instantiated. The original single-sample
benchmark wrapper was incompatible with the collator's batch-padding logic. Fixed by writing
a reusable `batch_evaluate(model, dataloader, device)` function that handles both the
benchmark and the full held-out evaluation, replacing two previously separate code paths.
Output format is identical to the original wrapper so downstream result-parsing code did not
need to change.

**Commits this week:**
- `feat: layer-freezing experiment — 5 variants (0–4 frozen layers) on SST-2 and IMDb`
- `feat: Pareto frontier plots — inference latency vs accuracy and training time vs F1`
- `feat: cross-dataset transfer evaluation — SST-2 model on IMDb, IMDb model on SST-2`
- `feat: error analysis — categorised error breakdown per model and dataset`
- `feat: full held-out evaluation on IMDb test set (25K) and SST-2 val set (872 samples)`
- `feat: batched inference loop for held-out evaluation, batch size 64`
- `fix: TorchScript JIT — switch to scripting path to resolve TracerWarning on dynamic padding`
- `docs: final report — metric equations, BiLSTM results, 9 references, updated timeline table`
- `refactor: consolidate all Week 3–4 experiments into final notebook with RUN_TRAINING flags`  
