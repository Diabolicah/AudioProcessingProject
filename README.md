# Explainable Auditory AI — Reconstructed Code

Working reconstruction of the code behind the project book **"Explainable
Auditory AI — Evaluating Auditory AI Hypotheses"** (Kozinets, Sayada, Elisha,
2025). The original code was lost; this repository restores a verified,
end-to-end pipeline that reproduces the book's results on the real corpora
(RAVDESS, TESS, CREMA-D).

**[REPRODUCING.md](REPRODUCING.md)** maps every table and figure in the book to
the command that regenerates it, records the reproduction scoreboard, and
documents every defect found and every reconstruction decision made.

## What this repository does

1. **Trains** the modified-EmoNet emotion classifier (ResNet + multi-head
   attention over mel-spectrograms) on RAVDESS, TESS and CREMA-D
   (`main.py`).
2. **Explains** it with Grad-CAM heat maps, masking and cross-correlation
   (`gradcam_utils.py`) and with TCAV concept scores over 12 synthetic
   spectrogram concepts (`concepts_creation.py`, `tcav_demo.py`).
3. **Analyses** the TCAV concept vectors: clustering metrics, per-label
   magnitudes, centroid assignment and PCA (`tcav_clustering.py`).

## Quick start

```bash
pip install -r requirements.txt          # or conda env create -f Models/SentimentAnalysis/project_env.yml
cd Models/SentimentAnalysis
python -m pytest tests -q                # 24 tests, no datasets needed

# full RAVDESS pipeline (datasets laid out per REPRODUCING.md section 1)
python main.py ravdess --augment                          # train  (~80% val, like the book)
python concepts_creation.py                                # the 12 concept datasets
python prob_vector.py ravdess --model <model.pt> --out pv.csv
python tcav_demo.py ravdess --attributes pv.csv --model <model.pt> \
    --out tcav.csv --multiclass-cavs --layer module3.blocks.1.conv2
python tcav_clustering.py --tcav tcav.csv --out-dir results/
python gradcam_utils.py --model <model.pt> --attributes pv.csv
```

## Reproduction status (details in REPRODUCING.md §6)

| result | book | this repo |
| --- | --- | --- |
| RAVDESS test accuracy | 72.2% | 74.3% (augmented) / 70.8% (plain) |
| RAVDESS validation accuracy | ~80% | 79.9% peak (augmented) |
| TESS test macro F1 | 0.995 | 1.000 |
| CREMA-D test macro F1 | 0.620 | 0.631 |
| per-class supports (Tables 3/8/10) | — | exact, all three datasets |
| concept dataset (Figures 5-6) | — | panel-for-panel match |
| PCA of concept space (Figure 15) | 79.8 / 14.6 | 81.4 / 14.7 |
| depth model (Table 2) | 0.705 test F1 | 0.579 (direction reproduced: worse than single-input) |

## Repository layout

```
Models/SentimentAnalysis/
    main.py                  training entry point (ravdess / depth / tess / cremad)
    models.py                ResNetWithAttention + training handler
    Preprocess*.py           mel-spectrogram pipeline and its constants
    audio_dataset.py         dataset adapters for the three corpora
    concept_defs.py          the 12 TCAV concepts (single source of truth)
    concepts_creation.py     synthetic concept-patch generator
    cav_classifier.py        concept classifiers (fixes captum's broken accuracy)
    tcav_demo.py             per-sample TCAV scoring (three CAV designs)
    tcav_clustering.py       clustering / PCA analysis -> the book's tables
    gradcam_utils.py         Grad-CAM sweep -> the book's figures
    reference/paper_tables.py  the book's published numbers, as data
    tests/                   24 regression tests pinning the published results
REPRODUCING.md               full reproduction guide and findings
```
