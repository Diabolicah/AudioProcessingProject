# Reproducing "Explainable Auditory AI — Evaluating Auditory AI Hypotheses"

Every table and figure in the project book, mapped to the command that produces
it. All commands run from `Models/SentimentAnalysis/`.

## 0. Environment

The original environment (`Models/SentimentAnalysis/project_env.yml`, conda,
Python 3.11):

```bash
conda env create -f Models/SentimentAnalysis/project_env.yml && conda activate final-project
```

`requirements.txt` at the repo root is the pip equivalent. **`requirements.yml`
at the repo root is stale** — it is a UTF-16 export of an unrelated environment
named `tf` and contains neither `captum` nor `grad-cam`, so the XAI half of the
project cannot be installed from it.

The whole pipeline also runs unchanged on a current stack (verified on Python
3.14 with torch 2.13, numpy 2.5, pandas 2.3, librosa 0.11, captum 0.9,
grad-cam 1.5.5). The original numpy 1.26 / torch 2.5 pins have no wheels for
3.14, so on a new machine either use conda with 3.11, or:

```bash
python -m venv .venv && .venv/Scripts/activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip install "pandas<3" scipy scikit-learn librosa==0.11.0 soundfile matplotlib seaborn tqdm pytest ipynbname captum grad-cam
```

Swap the torch index URL for `.../whl/cpu` if you do not need CUDA.

Check the setup without any datasets:

```bash
cd Models/SentimentAnalysis && python -m pytest tests -q
```

23 tests should pass; they pin the preprocessing constants, the label spaces,
the concept catalogue, the split, the model geometry and the CAV fixes below.

## 1. Data layout

The datasets are git-ignored. The code expects:

```
RAVDESS/RAVDESS DATASET/original_data/Actor_01/03-01-01-01-01-01-01.wav ...
RAVDESS/RAVDESS DATASET/neutral_synthesized/Actor_01/kids_rep1_act1.wav ...   # depth model only
CREMA-D/DATASETS/splitted data/{train,val,test}/*.wav
TESS/TESS_DATASET/{train,val,test}/*.wav
positive concepts/positive concepts dataset/<concept_name>/*.npy
```

RAVDESS is split in-process (70-10-20, stratified, seed 42). TESS and CREMA-D
ship as flat folders and are split on disk first:

```bash
python split_datasets.py cremad
python split_datasets.py tess
```

**The split seeds are part of the published result, not a free choice.** Verified
against the real corpora:

| dataset | seed | reproduces |
| --- | --- | --- |
| RAVDESS (in-process) | 42 | Table 3 supports exactly: 38/38/38/39/39/19/38/39 |
| CREMA-D | **123** | Table 10 supports exactly, all three splits |
| TESS | 42 | Table 8 supports: 280 / 40 / 80 per class |

Seed 42 on CREMA-D gives the same class proportions but hands the one spare
sample to a different class, so the val/test supports no longer line up with
Table 10. The seeds live in `SPLIT_JOBS` in `split_datasets.py`.

(Table 8 in the book lists val `surprised = 41`, which would make TESS total
2801; the corpus has exactly 2800 and the correct value is 40.)

## 2. Models — section 5.1.1-5.1.3, 5.2.1

| Command | Book artefacts |
| --- | --- |
| `python main.py ravdess` | Figures 7-9, Table 2 (single input), Table 3 (Sentence rows) |
| `python main.py ravdess --augment` | same, with 5x augmented training - the only run that reaches the book's ~80% validation accuracy |
| `python main.py depth` | Figures 10-12, Table 2 (depth), Table 3 (Depth rows) |
| `python main.py tess` | Figures 19-21, Tables 8-9 |
| `python main.py cremad` | Figures 22-24, Tables 10-11 |

Each run writes to
`training_results/<Model>/<RawData>/<Dataset>/<timestamp>/`:
loss / accuracy / confusion-matrix PNGs, `*-classification-report.csv` and
`.txt` (the per-class precision / recall / F1 tables), `training_info.txt`, and
the checkpoint.

The depth model additionally needs the XTTS neutral syntheses:

```bash
python neutral_gen.py            # writes the transcript files, then uncomment the TTS block
```

## 3. Grad-CAM — sections 4.3, 5.1.4

```bash
python prob_vector.py ravdess --model <ravdess.pt> --out ravdess_prob_vector.csv
python gradcam_utils.py --model <ravdess.pt> --attributes ravdess_prob_vector.csv
```

Produces Figure 4 (heat map), Figures 13-14 (heat map + 85% mask +
cross-correlation, per actor and emotion) under
`Benchmark_Results/Summary_By_Actor_New/<actor>/`.

`--min-confidence` defaults to 0.99, matching "data that was previously
classified with a 99% certainty by the SoftMax metrics". `--mask-quantile`
defaults to 0.85, matching "masking the top 85%".

Note that `INTENSITY_TO_INCLUDE = ['02']` in `gradcam_initilaization.py` keeps
only strong-intensity takes; RAVDESS records `neutral` at normal intensity only,
so neutral is excluded from the sweep by construction.

## 4. Concepts — section 4.4, Figures 5-6

```bash
python concepts_creation.py                # 12 dirs x 60 .npy patches, seed 42
python save_concepts_plots.py              # renders Figures 5-6
```

The 12 concepts are defined once, in `concept_defs.py`. Both the directory names
and the names TCAV resolves come from `dir_name()`, so they cannot drift apart.

The canonical spelling is `long_constant_thick` (underscores) — that is what the
generator writes and what the Figure 5/6 plot titles show. The saved outputs in
the analysis notebooks contain `long-constant-thick` (hyphens), so the concept
directories on the original machine had been renamed by hand after generation:
`concepts_creation.py` never produced hyphenated names, while the TCAV driver
looked them up. If you have hyphenated directories from that era, regenerate
them; `PreGeneratedConceptDataset` now raises instead of silently training a CAV
on zero positive examples.

## 5. TCAV — sections 4.4, 5.1.5, 5.2.3

```bash
# 1. per-sample probabilities (also gives the predicted label TCAV targets)
python prob_vector.py ravdess --model <ravdess.pt> --out ravdess_prob_vector.csv

# 2. per-sample TCAV scores + CAV accuracies
python tcav_demo.py ravdess --attributes ravdess_prob_vector.csv \
    --model <ravdess.pt> --out ravdess_tcav.csv

# 3. clustering, PCA and the magnitude tables
python tcav_clustering.py --tcav ravdess_tcav.csv --out-dir results/ravdess
```

Repeat with `tess` / `cremad`. **Pass a distinct `--model-id` per model** (the
default `<dataset>-<model stem>` already is distinct): Captum caches trained CAVs
under `./cav/<model_id>/` keyed only by concept id and layer, so a shared id
makes a later run silently reuse an earlier model's CAVs.

`tcav_clustering.py` writes:

| Output | Book artefacts |
| --- | --- |
| `clustering_all_concepts.csv` | Tables 4 (RAVDESS), 12 (TESS), 15 (CREMA-D) |
| `clustering_good_cavs.csv` | Tables 5, 13, 16 — CAV accuracy ≥ 85% |
| `average_magnitude_per_label.csv` | Tables 6, 14, 17 |
| `centroid_label_distribution.csv` | Tables 7, 18, 19 |
| `pca_all_concepts.png`, `pca_no_happy_neutral.png`, `pca_good_cavs.png` | Figures 15-17, 25-27, 29-31 |
| `cosine_silhouette.png` | Figures 18, 28, 32 |

The three `*.ipynb` notebooks contain the original, hand-run version of this
analysis. They still work; `tcav_clustering.load_tcav_csv` accepts both the
old space-separated column names (`"true label"`) and the underscore names
everything writes now.

## 6. Reproduction results (real corpora, 40 epochs, seed 42)

Run on RTX 5080 / torch 2.11+cu128. Splits reproduce the book's supports exactly
(see the seed table above).

### Final scoreboard

| paper artifact | book | this reproduction |
| --- | --- | --- |
| RAVDESS train (Table 2) | 1.000 | 1.000 |
| RAVDESS validation accuracy (Fig 8) | ~80% | 79.9% peak / 78.5% final (**with `--augment`**; plain runs plateau 72-76% across seeds) |
| RAVDESS test accuracy | 72.2% | 74.3% augmented / 70.8% plain |
| RAVDESS test weighted P/R/F1 (Table 2) | .728/.722/.721 | .748/.743/.738 augmented |
| TESS test macro F1 (Table 9) | 0.995 | 1.000 |
| CREMA-D test macro F1 (Table 11) | 0.620 | 0.631 |
| per-class supports (Tables 3/8/10) | — | exact, all three datasets |
| concepts (Figures 5-6) | — | panel-for-panel match (pinned by test) |
| Grad-CAM (Figures 4, 13-14) | — | 168 figures, same structure |
| CAV-accuracy regime (Tables 5/13/16) | straddles 0.85 | 0.835-0.896 (600 patches, legacy mode) |
| Table 6 magnitude range | -2.64..1.89 | -2.06..2.94 (legacy mode) |
| **PCA spectrum (Figure 15)** | **79.8 / 14.6** | **81.4 / 14.7** sim / 85.6 / 10.6 captum run (augmented model, multiclass CAVs; band ~80-86 / 10-15 over classifier seeds) |
| depth model (Table 2) | .713/.710/.705 | not run - needs the XTTS synthesis environment |

Two of the book's numbers are only reachable with **training-set augmentation**
(`main.py ravdess --augment`, using the transforms already present in
`audio_augmentation.py`): the ~80%% validation accuracy and the Figure 15 PCA
spectrum. The book does not mention augmentation, but the repository carries
`include_aug` flags and `*_noise.wav` filename patterns for it - lost wiring
this flag restores. Silence trimming (`--trim`), which the book *does* mention,
measurably does not help and slightly hurts test accuracy.

The Figure 15 reproduction selects three knobs the book leaves open - the conv
layer ("one of the convolution layers"), the CAV classifier's regularisation,
and the experimental-set design - by fit against the published spectrum. That
selection is reconstruction of a lost configuration, documented here so it is
not mistaken for independent confirmation.

### Classification — reproduces well

| model | metric | book | this run |
| --- | --- | --- | --- |
| RAVDESS single-input | test accuracy | 72.22% | **70.83%** |
| RAVDESS single-input | test weighted P/R/F1 | 0.728 / 0.722 / 0.721 | 0.712 / 0.708 / 0.695 |
| RAVDESS single-input | train weighted P/R/F1 | 1.000 | **1.000** |
| TESS | test macro P/R/F1 | 0.995 | **1.000** |
| CREMA-D | test macro P/R/F1 | 0.624 / 0.618 / 0.620 | 0.630 / 0.634 / 0.631 |
| CREMA-D | train macro | 1.000 | **1.000** |

The loss and accuracy curves match Figures 7-8 in shape: train saturates at 100%
by epoch 6-7, validation plateaus, and the overfitting the book notes is present.
RAVDESS validation plateaus ~73% here against ~80% in the book — the one
metric that sits meaningfully lower.

### Grad-CAM — reproduces

`gradcam_utils.py` on the trained RAVDESS model selects 525 recordings at the
book's 99% SoftMax threshold and renders 168 composite figures. They match
Figures 13-14 in structure and content: heat map concentrated at 2-4 kHz for
angry, harmonic striping in the 85% mask, cross-correlation map below.

### TCAV — reproduced after reconstructing the lost configuration

| quantity | book | this run | status |
| --- | --- | --- | --- |
| CAV accuracy regime | several concepts >= 0.85, others below | 0.835-0.896 with ~600 patches/concept | **reproduced** |
| magnitude scale (Table 6) | -2.64 … 1.89 | -2.06 … 2.94 (original config) | **reproduced** |
| PCA explained variance (Fig 15) | 79.8% / 14.6% | 81.4 / 14.7 (augmented model + multiclass CAVs; see below) | **reproduced** |
| per-concept magnitude profile | varies strongly per concept | model-specific; ours does not correlate with Table 6's exact values | not comparable |

The CAV-accuracy regime is reproduced only because the metric is broken: with
captum's classifier the reported number is the concept/random class balance, so
~600 patches against 100 randoms gives ~0.857 and lands either side of the 85%
bar. See `concept_defs.CONCEPT_SAMPLES_PER_DIR`.

**What does not reproduce, and what was ruled out.** The book's concept space is
genuinely two-dimensional — its Table 6, read as an 8x12 matrix, has explained
variance [80.1%, 16.4%], consistent with its own Figure 15 (79.8 / 14.6). Every
configuration reachable from the committed code is essentially rank-1:

| configuration | PCA PC1 / PC2 |
| --- | --- |
| **book, Figure 15** | **79.8 / 14.6** |
| original settings, 60 patches | 98.4 / 1.4 |
| original settings, 600 patches | 99.8 / 0.1 |
| corrected settings, 60 patches | 100.0 / 0.0 |
| original settings + z-scored PCA | 98.6 / 1.2 |
| original settings + L2-normalised PCA | 88.0 / 10.1 |

Ruled out as causes: captum version (the scoring code in `_tcav_sub_computation`
is byte-identical in 0.7.0 and 0.9.0), the two CAV defects, the patch count, and
the PCA preprocessing.

The measured cause is that the 12 CAV directions are nearly parallel — mean
pairwise cosine **0.995**. Every concept is a thin bright line and the negative
set is texture (solid / noise / dotted / stripes), so each classifier learns
"is there a line" rather than "rising vs dropping". A rank-1 concept matrix
follows necessarily. Swapping the negative set to the other 11 concepts makes
the directions near-orthogonal (mean cosine **-0.053**), which is what a
two-dimensional concept space needs.

The concept dataset itself is NOT the cause: the regenerated concepts match the
book's Figures 5-6 panel-for-panel (positions, lengths, slopes - pinned by
`test_concept_generation_reproduces_the_book_figures`).

**The design that recovers the book's PCA structure.** Sweeping CAV
experimental designs against cached layer gradients for all 1440 samples:

| CAV design | mean cos(CAV_i, CAV_j) | per-sample PCA PC1/PC2 |
| --- | --- | --- |
| **book, Figure 15** | ? | **79.8 / 14.6** |
| pairwise concept-vs-random (committed code) | +0.995 | 99.9 / 0.1 |
| pairwise concept-vs-random-lines | +0.86 | 94.8 / 2.7 |
| **one multiclass classifier, 12 concepts + random** | **~0.00** | **73.6 / 9.6** |
| multiclass, 12 concepts without random | -0.06 | 45-56 / 14-25 |

Only the multiclass family lands in the book's regime, and captum supports it
natively (an experimental set may hold all 13 concepts). It is available as
`tcav_demo.py --multiclass-cavs`.

**The honest tension that remains.** The multiclass fit also shows the 12
concepts are not mutually discriminable at this layer - 13-class accuracy is at
chance (0.08-0.13) with every solver tried, because e.g. steep-thick vs
steep-thin differ by ~2 px of Gaussian width. So no single configuration can
simultaneously produce (a) real per-concept accuracies straddling 85% and
(b) the two-dimensional concept space of Figure 15. The book's numbers are
jointly consistent only with the broken class-balance "accuracy" (which with
~600 patches per concept straddles 85% by construction) alongside
multiclass-style magnitudes. Since the original code is lost, the exact
combination cannot be recovered; `--multiclass-cavs`, `--raw-concept-scale`,
`--legacy-captum-classifier` and the 600-patch generator span every candidate.

## 7. Two defects that invalidated the CAV quality numbers

Both were found by running the pipeline end to end and both are fixed, but they
mean the **"Good Cavs" tables (5, 13, 16) in the book were computed from a
meaningless metric**, and the CAVs themselves were fitted to activations no real
spectrogram produces. Re-run the TCAV stage before quoting those tables.

**a. Concept patches were on the wrong input scale.**
`power_to_db(..., ref=np.max)` puts real spectrograms in `[-80, 0]` dB, but
`concepts_creation` normalises the synthetic patches to `[0, 1]`. The network's
BatchNorm layers are calibrated for the dB range, so `[0, 1]` patches land far
outside it and concept/random activations collapse together. Measured on a
trained classifier:

| concept patches fed as | CAV accuracy (12 concepts) |
| --- | --- |
| `[0, 1]`, as originally written | 0.623 — exactly the majority baseline, i.e. nothing learned |
| mapped to `[-80, 0]` dB | 1.000 |

Fixed by `PreprocessParams.patch_to_db_scale`, applied to the positive **and**
random concepts. `tcav_demo.py --raw-concept-scale` restores the old behaviour.

**b. Captum's default concept classifier does not report an accuracy.**
`DefaultClassifier` wraps a scikit-learn `SGDClassifier`; for a binary
concept-vs-random problem `coef_` has one row, so the wrapped module emits one
column and `argmax(predict, dim=1)` is always `0`. It always predicts the first
class, and the reported `accs` is just the proportion of positives in the random
test split. On perfectly separable data captum 0.9.0 returns **0.25–0.40** where
the true accuracy is **1.00**.

Fixed by `cav_classifier.BinaryConceptClassifier`, which does a stratified
split, reports real accuracy, also reports `majority_baseline` for comparison,
and returns the two CAV rows TCAV indexes by concept.

## 8. Things the book describes that the code reads differently

* **"Volume standardization"** (section 3.2) is performed by
  `librosa.power_to_db(..., ref=np.max)`, which puts every recording's peak at
  0 dB. `PreprocessParams.SPECTROGRAM_NORMALIZATION` is an *additional* hook,
  identity by default — that is the behaviour every existing checkpoint was
  trained with. Changing it invalidates trained models.
* **Figure 3** draws the residual block with 2×2 convolutions; the code (and
  Figure 1) uses 3×3. The code follows Figure 1.
* **Class weighting.** Training uses `CrossEntropyLoss(weight=N/count)`, which
  the book does not mention. With `reduction='mean'` PyTorch divides by the sum
  of weights, so the loss magnitude still starts near ln(8) ≈ 2.08 as Figure 7
  shows — the weighting is consistent with the reported curves, just undocumented.
* **Cross-correlation kernels** exist for all seven non-neutral emotions in
  `correlation_kernel_playground.py`, but four of them (`calm`, `sad`,
  `disgusted`, `fearful`) are the same flat 3×15 placeholder, so their
  correlation maps carry no emotion-specific information. Section 5.1.4 already
  reports the cross-correlation attempt as inconclusive; this is why.
