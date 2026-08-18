# Fixes, Changes and Reproduction Results

Everything done to restore the lost code behind *"Explainable Auditory AI —
Evaluating Auditory AI Hypotheses"* and to reproduce its published results on
the real corpora. Companion to [REPRODUCING.md](REPRODUCING.md), which maps
each table/figure to the command that regenerates it.

## 1. Correctness bugs fixed (silently wrong results)

| # | Bug | Effect before the fix |
|---|---|---|
| 1 | TCAV looked up **hyphenated** concept names (`long-constant-thick`) while the generator wrote **underscores**; a `mkdir` in the dataset class hid the mismatch | CAVs trained on **zero positive examples** without any error |
| 2 | Captum's `DefaultClassifier` always predicts the first class for binary concept-vs-random sets (`argmax` over a one-column output) — verified in captum **0.7.0** (the authors' version) and 0.9.0 | The reported `cav_acc` was the test-split class balance, not an accuracy; the ≥85% "Good CAVs" filter (Tables 5/13/16) selected on noise |
| 3 | Concept patches fed to the model as `[0,1]` while real spectrograms are `[-80,0]` dB | CAV probe accuracy at chance (0.50); fixed → 1.00 |
| 4 | Train/val/test split computed from a Python `set` (per-process iteration order) | A different split on every run despite `random_state=42` |
| 5 | A separate `LabelEncoder` fitted per split | A split missing one class encodes every later class to a different integer — scrambled labels |
| 6 | `'disgust'` vs `'disgusted'` spelling split across modules | Two different classes downstream; the disgust cross-correlation kernel never matched its label |
| 7 | Leftover `df.head(10)` in the TCAV driver | TCAV scored **10 samples** instead of all of them |
| 8 | `label_2_index.get(name)` → `None` passed as the TCAV target | Silently scored the wrong class instead of raising |
| 9 | Captum's CAV cache keyed only by (concept ids, layer) with a shared model id | A TESS run silently reused CAVs trained on the RAVDESS model |
| 10 | FC input size computed as `n // 8` instead of repeated ceil-halving | Shape crash for any input size not divisible by 8 |
| 11 | `file_name = X if None else file_name` (always False) | Every custom plot filename silently discarded |
| 12 | Test-label buffers appended on every call | Confusion-matrix counts doubled when test ran twice |
| 13 | RAVDESS neutral-file glob `*-*-01-*` matched `-01-` anywhere in the name | Non-neutral takes selected as "neutral" references |
| 14 | CREMA-D split seed: my earlier standardisation to 42 was wrong — the published Table 10 supports need the original **123** | Val/test supports off by one sample per class |
| 15 | `random_thickness_boundary` accepted but never forwarded by `create_concept_dir` | Two concept specs' parameter ignored — **kept intentionally**: the published concepts (Figures 5–6) embed this behaviour, and a test now pins it |

## 2. Crashes and unrunnable code fixed

- `gradcam_utils.py` ran its whole sweep **at import time** with `Models.SentimentAnalysis.*` imports that don't resolve; now a function + CLI
- `split_datasets.py` copied the entire CREMA-D corpus at import; `correlation_kernel_playground.py` opened a blocking plot window at import
- `gradcam_initilaization.py` loaded a checkpoint at import time from a hard-coded relative path
- `ndarray.ptp()` (removed in NumPy 2), cp1252 console crashes on `→`/`↪` characters, undefined names in dead code paths
- `neutral_gen.py` required the TTS package just to import; now optional
- Broken UTF-16 `requirements.yml` (an unrelated env without captum/grad-cam) → working `requirements.txt`

## 3. Reconstruction of the lost configuration

The committed repo did not reproduce two published results; both required
reconstructing configuration the paper never states. Selection was done by fit
against the published numbers and is documented as such.

- **`main.py ravdess --augment`** — 5× training-set augmentation (noise, pitch ±2, stretch ×1.1, using the repo's own `audio_augmentation.py`, whose `include_aug` wiring was vestigial). Required for the ~80% validation accuracy (plain runs plateau at 72–76% across seeds) *and* for the Figure 15 PCA.
- **`tcav_demo.py --multiclass-cavs --layer module3.blocks.1.conv2`** — one GPU cross-entropy classifier over all concepts instead of 12 pairwise fits. The only CAV design whose concept space has rank > 1: pairwise CAV directions have mean pairwise cosine **+0.995** (every classifier learns "is there a line"), multiclass ≈ 0. Grid-searched over layer × loss × weight decay × set design.
- **Legacy flags** `--raw-concept-scale --legacy-captum-classifier` reproduce the book's original configuration exactly (including its broken accuracy metric), for auditability.
- **`--trim`** implements the silence-trimming the book describes (§3.2) — measured: no validation gain, test drops. Off by default.
- **`main.py depth --align-durations`** — time-stretch the synthesized neutral channel onto each recording's duration. XTTS speaks ~31% faster than the RAVDESS actors *whatever its settings*, so without this the depth model's two channels are effectively unrelated in time (measured inter-channel temporal correlation 0.076; 0.260 after). Worth +0.036 test F1. Not in the book, but the book's own hypothesis — that the network can "identify the differences between the two input" — presupposes the channels are comparable.

## 4. New infrastructure

`concept_defs.py` (single source of truth for the 12 concepts) ·
`cav_classifier.py` (honest binary + GPU multiclass CAV classifiers) ·
`tcav_clustering.py` (regenerates Tables 4–7 / 12–19 and the PCA figures) ·
classification-report export in `models.py` (Tables 2/3/8–11) ·
`reference/paper_tables.py` (the book's numbers as data) ·
`tests/` (24 regression tests, no datasets needed) ·
CLIs for every stage · `README.md` · `REPRODUCING.md`.

## 5. Final numbers vs the paper

Real corpora, 40 epochs, seed 42, RTX 5080. All splits reproduce the book's
per-class supports exactly (RAVDESS/TESS seed 42, CREMA-D seed 123).

| result | paper | this reproduction |
|---|---|---|
| RAVDESS train P/R/F1 (Table 2) | 1.000 | **1.000** |
| RAVDESS validation accuracy (Fig 8) | ~80% | **79.9% peak** / 78.5% final (augmented); 72–76% plain |
| RAVDESS test accuracy | 72.2% | **74.3%** augmented / 70.8% plain |
| RAVDESS test weighted P/R/F1 (Table 2) | .728 / .722 / .721 | **.748 / .743 / .738** |
| TESS test macro P/R/F1 (Table 9) | 0.995 | **1.000** |
| CREMA-D test macro P/R/F1 (Table 11) | 0.620 | **0.631** |
| per-class supports (Tables 3 / 8 / 10) | — | **exact**, all three datasets |
| concept dataset (Figures 5–6) | — | **panel-for-panel match** (pinned by test) |
| Grad-CAM (Figures 4, 13–14) | — | **reproduced** — 168 figures, same structure |
| CAV-accuracy regime (Tables 5/13/16) | straddles 0.85 | **0.835–0.896** (600 patches, legacy mode) |
| Table 6 magnitude range | −2.64 … 1.89 | **−2.06 … 2.94** (legacy mode) |
| PCA of concept space (Figure 15) | 79.8 / 14.6 | **81.4 / 14.7** (augmented model + multiclass CAVs; captum end-to-end run 85.6 / 10.6; varies ~80–86 / 10–15 over classifier seeds) |
| depth model test P/R/F1 (Table 2) | .713 / .710 / .705 | **.627 / .618 / .615** with `--align-durations` (.582/.587/.579 without); train 1.000 exact; direction reproduced — worse than single-input |

Book errata found on the way: Table 8 lists TESS val `surprised = 41` (total
2801 vs the corpus's 2800; correct value 40); Table 3's validation supports sum
to 143, not 144.

## 6. Verification state

24/24 regression tests · pyflakes-clean · all 23 modules import · both TCAV
modes smoke-tested after cleanup · every recorded number re-computed unchanged.
History: 7 commits pushed to `main`, `d47b0dc` through `c1359ca`.
