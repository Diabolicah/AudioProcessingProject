"""The 12 TCAV concepts of the project book (section 4.4, Figures 5-6).

Kept in its own dependency-free module so that anything needing the concept
*names* - TCAV, the multi-label tagger, the analysis notebooks, the tests - does
not have to import librosa and matplotlib through `concepts_creation`.

`CONCEPT_UNIQUE_NAMES` is the single source of truth for the names. It is
derived from `CONCEPT_SPECS` through the same `dir_name()` used to write the
directories on disk, so the names TCAV looks up can never drift away from the
names the generator produces.
"""

from typing import List

# Sample count and RNG seed used to generate every concept directory.
#
# The book does not state how many patches per concept were used. 60 is what the
# committed generator wrote, but the published CAV accuracies (Tables 5/13/16,
# which keep concepts at ">= 85% confidence") can only arise from ~570-600.
# Captum's default concept classifier reports the test-split class balance
# rather than an accuracy (see cav_classifier.py), so with P concept patches
# against RANDOM_CONCEPT_SAMPLES=100 randoms it reports P/(P+100):
#
#     60 patches  -> 0.375   (measured 0.396; nothing passes the 85% bar)
#    600 patches  -> 0.857   (measured 0.835-0.896; some pass, some do not,
#                             which is exactly what the book's tables show)
#
# Use 600 to reproduce the book's CAV-accuracy regime; 60 is fine for the
# corrected pipeline, where every concept scores 1.000 either way.
CONCEPT_SAMPLES_PER_DIR = 60
CONCEPT_SAMPLES_PAPER_REGIME = 600
CONCEPT_RANDOM_SEED = 42


def dir_name(length: str, tone: str, rate: str, thickness: str) -> str:
    """Canonical directory name for a concept. `rate` is dropped when constant."""
    return f"{length}_{tone}_{rate}_{thickness}" if tone != "constant" else f"{length}_{tone}_{thickness}"


CONCEPT_SPECS: List[dict] = [
    # ---- rising ----
    dict(length="long",  tone="rising",   rate="steep", thickness="thin",
         mean_length=0.085, std_length=0.01,  mean_tone_degree=57,  std_tone_degree=8, mean_thickness=3),
    dict(length="long",  tone="rising",   rate="steep", thickness="thick",
         mean_length=0.085, std_length=0.007, mean_tone_degree=57,  std_tone_degree=4, mean_thickness=4.9,
         random_thickness_boundary=0.2),
    dict(length="long",  tone="rising",   rate="flat",  thickness="thick",
         mean_length=0.1,   std_length=0.01,  mean_tone_degree=33,  std_tone_degree=8, mean_thickness=3.3),
    dict(length="short", tone="rising",   rate="steep", thickness="thin",
         mean_length=0.049, std_length=0.004, mean_tone_degree=32,  std_tone_degree=7, mean_thickness=3),
    dict(length="short", tone="rising",   rate="steep", thickness="thick",
         mean_length=0.05,  std_length=0.008, mean_tone_degree=32,  std_tone_degree=7, mean_thickness=5),
    # ---- constant ----
    dict(length="long",  tone="constant", rate="ignored", thickness="thick",
         mean_length=0.4,   std_length=0.06,  mean_tone_degree=0,   std_tone_degree=1.3, mean_thickness=2.92),
    dict(length="short", tone="constant", rate="ignored", thickness="thick",
         mean_length=0.17,  std_length=0.027, mean_tone_degree=0,   std_tone_degree=1.3, mean_thickness=2.92),
    # ---- dropping ----
    dict(length="long",  tone="dropping", rate="steep", thickness="thin",
         mean_length=0.085, std_length=0.01,  mean_tone_degree=-57, std_tone_degree=8, mean_thickness=3),
    dict(length="long",  tone="dropping", rate="steep", thickness="thick",
         mean_length=0.085, std_length=0.007, mean_tone_degree=-57, std_tone_degree=4, mean_thickness=4.9,
         random_thickness_boundary=0.2),
    dict(length="long",  tone="dropping", rate="flat",  thickness="thick",
         mean_length=0.1,   std_length=0.01,  mean_tone_degree=-33, std_tone_degree=8, mean_thickness=3.3),
    dict(length="short", tone="dropping", rate="steep", thickness="thin",
         mean_length=0.049, std_length=0.004, mean_tone_degree=-32, std_tone_degree=7, mean_thickness=3),
    dict(length="short", tone="dropping", rate="steep", thickness="thick",
         mean_length=0.05,  std_length=0.008, mean_tone_degree=-32, std_tone_degree=7, mean_thickness=5),
]

# Sorted so the order matches both a pandas `pivot_table(columns="concept_name")`
# and the column order of Tables 6, 14 and 17.
CONCEPT_UNIQUE_NAMES: List[str] = sorted(
    dir_name(spec["length"], spec["tone"], spec["rate"], spec["thickness"])
    for spec in CONCEPT_SPECS
)
