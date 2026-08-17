"""TCAV scoring for every trained emotion classifier (project book, section 4.4).

For each sample in a probability-vector CSV this computes, per concept, the
Captum TCAV `sign_count` and `magnitude`, plus the accuracy of the linear CAV
classifier. The per-sample table feeds `tcav_clustering.py`, which regenerates
Tables 4-7, 12-19 and the PCA figures.

    python tcav_demo.py ravdess --attributes <prob_vector.csv> --model <model.pt> --out <tcav.csv>
    python tcav_demo.py tess    ...
    python tcav_demo.py cremad  ...

Three CAV designs are available (full comparison in REPRODUCING.md section 6):

  default            12 pairwise concept-vs-random classifiers, dB-scaled
                     patches, honest accuracies (BinaryConceptClassifier).
                     Every CAV scores 1.000 but the 12 directions are
                     near-parallel, so the concept space is rank-1.

  --multiclass-cavs  one classifier over all concepts (+ random unless
                     --no-random-in-set), GPU cross-entropy fit. The only
                     design whose per-sample PCA matches the book's Figure 15;
                     combine with `--layer module3.blocks.1.conv2` and a model
                     trained with `main.py ravdess --augment` for the closest
                     reproduction (81.4/14.7 vs the book's 79.8/14.6).

  --raw-concept-scale --legacy-captum-classifier
                     the configuration the book's tables were produced under:
                     [0,1] patches and captum's DefaultClassifier, whose
                     reported "accuracy" is the class balance (with ~600
                     patches per concept it straddles the 85% bar exactly as
                     Tables 5/13/16 require).
"""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from captum.concept import TCAV, Concept
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ConstPaths import conceptPaths
from Preprocess import audio_to_mel_spectrogram
from PreprocessParams import (CREMAD_LABELS, FREQUENCY_BIN_COUNT, RAVDESS_LABELS,
                              TARGET_FRAMES, TESS_LABELS, index_to_label, label_to_index,
                              patch_to_db_scale)
from cav_classifier import BinaryConceptClassifier, TorchLinearConceptClassifier
from concept_defs import CONCEPT_UNIQUE_NAMES
from concepts_creation import generate_random_pattern_spectrogram

# Convolution the CAVs are trained in. Any layer of module3 works; this is the
# one every reported result used.
TCAV_LAYER = "module3.blocks.0.conv2"

# Number of random negative patches. Captum trains the linear concept classifier
# on positives-vs-random, so this should match the per-concept sample count.
RANDOM_CONCEPT_SAMPLES = 100

# Book, section 5.1.5: a CAV counts as "good" when its linear classifier reaches
# at least 85% accuracy. Used by the notebooks to build the "Good Cavs" tables.
GOOD_CAV_ACCURACY = 0.85

# Map the [0, 1] concept patches onto the [-80, 0] dB range the model actually
# sees. See PreprocessParams.patch_to_db_scale - without this the CAVs are
# fitted to activations no real spectrogram ever produces. Set False only to
# reproduce the original, unscaled behaviour.
SCALE_CONCEPTS_TO_DB = True

# Label space per dataset, in the order the model's output indices follow.
DATASET_LABELS = {
    "ravdess": RAVDESS_LABELS,
    "tess": TESS_LABELS,
    "cremad": CREMAD_LABELS,
}

# Kept for backwards compatibility with the existing notebooks, which import
# LABEL_EMOTION_MAPPING from this module. It is derived from the canonical label
# list rather than hand-written, so it can no longer disagree with the encoder.
LABEL_EMOTION_MAPPING = index_to_label(RAVDESS_LABELS)


# PyTorch Datasets for TCAV

class PreGeneratedRandomSpectrogramDataset(Dataset):
    """
    PyTorch Dataset that pre-generates all random spectrogram in memory.
    """

    def __init__(self, n_samples: int, freq_count = FREQUENCY_BIN_COUNT, frames = TARGET_FRAMES,
                 rng_seed: Optional[int] = 42, device: Optional[torch.device] = None,
                 scale_to_db: bool = SCALE_CONCEPTS_TO_DB):
        self.n_samples = n_samples
        self.freq_count = freq_count
        self.frames = frames
        self.rng = np.random.default_rng(rng_seed)

        # Pre-generate all spectrograms in memory
        patches = np.array([generate_random_pattern_spectrogram(freq_count, frames, rng=self.rng)
                     for _ in range(n_samples)])
        # The same scaling must be applied to the positives, otherwise the
        # concept classifier separates the two sets on scale alone.
        if scale_to_db:
            patches = patch_to_db_scale(patches)
        self.data = torch.tensor(patches, dtype=torch.float32)
        if device is not None:
            self.data = self.data.to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Ensure shape [1, H, W] per sample
        x = self.data[idx]
        return x.unsqueeze(0)

    @property
    def get_data(self):
        return self.data

class PreGeneratedConceptDataset(Dataset):
    """
    PyTorch Dataset that loads the pre-generated patches for one concept.
    """

    def __init__(self, concept_name: str, root_concept_dir: Path = conceptPaths.ALL_CONCEPTS,
                 freq_count = FREQUENCY_BIN_COUNT, frames_count = TARGET_FRAMES,
                 device: Optional[torch.device] = None,
                 scale_to_db: bool = SCALE_CONCEPTS_TO_DB):
        self.concept_name = concept_name
        self.root_concept_dir = Path(root_concept_dir)
        self.freq_count = freq_count
        self.frames = frames_count

        concept_dir = self.root_concept_dir / self.concept_name
        # Fail loudly. The previous version called `concept_dir.mkdir(...)`
        # here, so a misspelled concept name silently created an empty
        # directory and TCAV went on to train a CAV on zero positive examples.
        if not concept_dir.is_dir():
            raise FileNotFoundError(
                f"Concept directory {concept_dir} does not exist. "
                f"Run `python concepts_creation.py` to generate the 12 concept datasets."
            )

        patches = [np.load(npy_file) for npy_file in sorted(concept_dir.glob("*.npy"))]
        if not patches:
            raise ValueError(f"Concept directory {concept_dir} contains no .npy patches.")

        expected = (freq_count, frames_count)
        if patches[0].shape != expected:
            raise ValueError(
                f"Concept '{concept_name}' patches have shape {patches[0].shape}, "
                f"but the model expects {expected}. Regenerate the concepts after "
                f"changing MAX_SPECTOGRAM_DURATION_IN_SECONDS or FREQUENCY_BIN_COUNT."
            )

        self.n_samples = len(patches)
        stacked = np.array(patches)
        # Concept patches are saved normalised to [0, 1]; real model inputs are
        # dB in [-80, 0]. See PreprocessParams.patch_to_db_scale.
        if scale_to_db:
            stacked = patch_to_db_scale(stacked)
        self.data = torch.tensor(stacked, dtype=torch.float32)
        if device is not None:
            self.data = self.data.to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Ensure shape [1, H, W] per sample
        x = self.data[idx]
        return x.unsqueeze(0)

    @property
    def get_data(self):
        return self.data

# Functions

def build_experimental_sets(positive_concepts: List[Concept], random_concept: Concept,
                            multiclass: bool, include_random: bool = True) -> List[List[Concept]]:
    """Pairwise: one [concept, random] set per concept (the committed design).

    Multiclass: a single set holding all 12 concepts plus random, so one
    classifier is trained over 13 classes and every concept's CAV row must be
    distinct from the others by construction. This is the only design measured
    to produce a concept space of rank > 1 (see REPRODUCING.md section 6): with
    pairwise fitting the 12 CAV directions are near-parallel (mean pairwise
    cosine 0.995 - every classifier learns "is there a line") and the per-sample
    concept vectors collapse to one PCA component, where Figure 15 of the book
    shows two.
    """
    if multiclass:
        members = positive_concepts + ([random_concept] if include_random else [])
        return [members]
    return [[c, random_concept] for c in positive_concepts]


def init_tcav(model_path: Path, model_id: str, device: Optional[torch.device] = None,
              layer: str = TCAV_LAYER, cav_save_path: str = "./cav/",
              scale_to_db: bool = SCALE_CONCEPTS_TO_DB,
              legacy_classifier: bool = False,
              multiclass: bool = False,
              include_random: bool = True) -> dict:
    """Load a trained classifier and build the TCAV object plus its concepts.

    `model_id` scopes the CAV cache. Captum stores trained CAVs under
    `<cav_save_path>/<model_id>/` and reuses them when `force_train=False`, and
    the cache key is only (concept ids, layer) - so without a per-run model_id
    a TESS run would silently reuse the CAVs trained on the RAVDESS model.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    # `classifier=BinaryConceptClassifier()` rather than captum's default: the
    # default reports the test-split class balance instead of an accuracy for
    # binary concept-vs-random problems (see cav_classifier.py), which is the
    # number the book's "Good Cavs" tables filter on.
    # `legacy_classifier=True` restores captum's default, which together with
    # `scale_to_db=False` reproduces the exact configuration the book's figures
    # were produced under.
    if legacy_classifier:
        classifier = None
    elif multiclass:
        # GPU cross-entropy fit with weight decay 1e-3: the grid-search winner
        # that reproduces the book's Figure 15 spectrum (78.7/11.0 vs 79.8/14.6
        # at this layer, concepts-only set - see REPRODUCING.md).
        classifier = TorchLinearConceptClassifier(weight_decay=1e-3)
    else:
        classifier = BinaryConceptClassifier()
    tcav = TCAV(model, [layer], model_id=model_id, save_path=cav_save_path,
                classifier=classifier, test_split_ratio=0.33)

    positive_concepts: List[Concept] = [
        Concept(id=concept_idx, name=concept_name,
                data_iter=DataLoader(PreGeneratedConceptDataset(concept_name=concept_name, device=device,
                                                                scale_to_db=scale_to_db),
                                     shuffle=False))
        for concept_idx, concept_name in enumerate(CONCEPT_UNIQUE_NAMES)
    ]

    # This concept is the negative of concepts.
    negative_concept_dataset = PreGeneratedRandomSpectrogramDataset(
        n_samples=RANDOM_CONCEPT_SAMPLES, freq_count=FREQUENCY_BIN_COUNT,
        frames=TARGET_FRAMES, device=device, scale_to_db=scale_to_db)
    random_concept = Concept(id=len(positive_concepts), name='random',
                             data_iter=DataLoader(negative_concept_dataset, shuffle=False))

    return {'tcav': tcav, 'positive-concepts': positive_concepts,
            'random-concept': random_concept, 'layer': layer, 'device': device,
            'experimental-sets': build_experimental_sets(positive_concepts, random_concept,
                                                         multiclass, include_random)}


# Backwards-compatible alias for the original (misspelled) name.
init_tcav_with_pamalia_dict = init_tcav


def _compute_cav_accuracy_df(tcav: TCAV,
                             positive_concepts: List[Concept],
                             random_concept: Concept,
                             experimental_sets: Optional[List[List[Concept]]] = None,
                             float_precision: int = 3) -> pd.DataFrame:
    """
    Trains / loads CAVs once and extracts the linear concept-classifier accuracy
    per (concept, layer). Returns a DataFrame with columns:
    [concept_name, layer_name, cav_acc]

    For a multiclass experimental set captum stores one stats dict for the whole
    CAV; the per-concept number then comes from `per_class_accs` (added by
    BinaryConceptClassifier) and falls back to the shared overall accuracy.
    """
    if experimental_sets is None:
        experimental_sets = [[c, random_concept] for c in positive_concepts]

    # Train / load CAVs for all concepts & layers in one shot
    cavs_dict = tcav.compute_cavs(experimental_sets, force_train=False)

    rows = []
    # cavs_dict maps "<id>-<id>-..." -> {layer_name: CAV}
    for concepts_key, layer_map in cavs_dict.items():
        try:
            set_ids = [int(t) for t in str(concepts_key).split("-")]
        except Exception:
            continue

        for layer_name, cav_obj in layer_map.items():
            if cav_obj is None or cav_obj.stats is None:
                continue
            acc = cav_obj.stats.get("accs", None)  # DefaultClassifier returns {"accs": <tensor/float>}
            if isinstance(acc, torch.Tensor):
                acc = acc.detach().cpu().item()
            per_class = cav_obj.stats.get("per_class_accs", None)
            classes = cav_obj.stats.get("classes", set_ids)
            if isinstance(classes, torch.Tensor):
                classes = classes.tolist()

            for cid in set_ids:
                if not (0 <= cid < len(positive_concepts)):
                    continue        # the random concept has no accuracy row
                concept_acc = acc
                if per_class is not None and cid in classes:
                    concept_acc = float(per_class[classes.index(cid)])
                rows.append({
                    "concept_name": positive_concepts[cid].name,
                    "layer_name": layer_name,
                    "cav_acc": round(float(concept_acc), float_precision) if concept_acc is not None else np.nan,
                })

    return pd.DataFrame(rows, columns=["concept_name", "layer_name", "cav_acc"])


def _tcav_dict_per_sample_to_df(tcav_raw_dict: dict, scores_by_sample: dict, concept_names: list[str],
                                float_precision: int = 3) -> pd.DataFrame:
    """
    Flatten Captum TCAV results into a DataFrame with columns:
    ["path", "concept_name", "layer_name", "positive_percentage", "magnitude", "cav_acc"]
    """
    rows = []
    for path, exp_sets in scores_by_sample.items():
        # exp_key is the '-'-joined concept ids of one experimental set:
        # "0-12" for a pairwise [concept, random] set, "0-1-...-12" for the
        # multiclass set. metrics tensors are ordered like the set, so entry j
        # belongs to the j-th id; ids >= len(concept_names) are the random
        # baseline and carry no concept row.
        for exp_key, layer_dict in exp_sets.items():
            try:
                set_ids = [int(t) for t in str(exp_key).split("-")]
            except Exception:
                continue  # skip malformed keys

            # Usually there's a single chosen layer, but handle multiple layers just in case
            for layer_name, metrics in layer_dict.items():
                sc = metrics.get("sign_count")
                mg = metrics.get("magnitude")
                if sc is None or mg is None:
                    continue

                # Convert torch tensors to Python floats
                if isinstance(sc, torch.Tensor):
                    sc = sc.detach().cpu().tolist()
                if isinstance(mg, torch.Tensor):
                    mg = mg.detach().cpu().tolist()

                for j, cid in enumerate(set_ids):
                    if not (0 <= cid < len(concept_names)):
                        continue
                    rows.append({
                        "path": path,
                        "concept_name": concept_names[cid],
                        "layer_name": layer_name,
                        "positive_percentage": round(float(sc[j]), float_precision),
                        "magnitude": round(float(mg[j]), float_precision),
                    })
    per_sample_df = pd.DataFrame(rows, columns=[
        "path", "concept_name", "layer_name", "positive_percentage", "magnitude"
    ])

    acc_df = _compute_cav_accuracy_df(tcav=tcav_raw_dict['tcav'],
                                      positive_concepts=tcav_raw_dict['positive-concepts'],
                                      random_concept=tcav_raw_dict['random-concept'],
                                      experimental_sets=tcav_raw_dict.get('experimental-sets'))
    # merge each row of acc_df with every row in per_sample_df that has the same concept and layer
    return per_sample_df.merge(acc_df, on=["concept_name", "layer_name"], how="left")


def _get_tcav_dict_per_sample(tcav_raw_dict: dict, all_filtered_data: pd.DataFrame,
                              label_2_index: dict) -> dict:
    """Run `tcav.interpret` once per sample, targeting that sample's predicted class."""
    tcav = tcav_raw_dict['tcav']
    positive_concepts = tcav_raw_dict['positive-concepts']
    random_concept = tcav_raw_dict['random-concept']
    device = tcav_raw_dict['device']

    experimental_sets = tcav_raw_dict.get('experimental-sets') or \
        [[c, random_concept] for c in positive_concepts]
    tcav_dict_per_sample = {}

    for _, row in tqdm(all_filtered_data.iterrows(), total=len(all_filtered_data), desc="Processing samples"):
        label_name = row['predicted_label']
        path = row['path']

        if label_name not in label_2_index:
            # Previously `label_2_index.get(...)` returned None here and was
            # passed straight to `target=`, which silently scored against the
            # wrong class instead of failing.
            raise KeyError(
                f"Predicted label {label_name!r} for {path} is not in the declared label "
                f"space {sorted(label_2_index)}. Check that the probability-vector CSV and "
                f"the --dataset argument refer to the same model."
            )
        label_index = label_2_index[label_name]

        sample = torch.tensor(audio_to_mel_spectrogram(Path(path)), dtype=torch.float32)
        sample = sample.unsqueeze(0).unsqueeze(0).to(device)  # shape [1, 1, H, W]

        tcav_dict_per_sample[path] = tcav.interpret(
            inputs=sample,
            experimental_sets=experimental_sets,
            target=label_index,
        )

    return tcav_dict_per_sample


def get_tcav_per_sample(attribute_csv_path: Path, model_path: Path, label_2_index: dict,
                        model_id: str, limit: Optional[int] = None,
                        scale_to_db: bool = SCALE_CONCEPTS_TO_DB,
                        legacy_classifier: bool = False,
                        multiclass: bool = False,
                        include_random: bool = True,
                        layer: str = TCAV_LAYER) -> pd.DataFrame:
    """Per-sample TCAV table joined onto the probability-vector attributes.

    :param limit: Debug aid - score only the first N samples. Leave as None for
        the reported results; an accidental `head(10)` here is what made an
        earlier version report scores over ten recordings.
    """
    df_attributes = pd.read_csv(attribute_csv_path)
    df_attributes = normalize_tcav_columns(df_attributes)

    if limit is not None:
        print(f"[WARN] --limit {limit}: scoring only the first {limit} samples, results are NOT the reported ones.")
        df_attributes = df_attributes.head(limit)

    # drop the per-class probability columns; only path / true / predicted are needed
    df_attributes = df_attributes.drop(columns=df_attributes.filter(regex=r'^prob[ _]').columns)

    tcav_raw_dict = init_tcav(model_path=model_path, model_id=model_id, scale_to_db=scale_to_db,
                              legacy_classifier=legacy_classifier, multiclass=multiclass,
                              include_random=include_random, layer=layer)

    tcav_proccessed_dict = _get_tcav_dict_per_sample(tcav_raw_dict=tcav_raw_dict,
                                                     all_filtered_data=df_attributes,
                                                     label_2_index=label_2_index)

    df_tcav = _tcav_dict_per_sample_to_df(tcav_raw_dict=tcav_raw_dict,
                                          scores_by_sample=tcav_proccessed_dict,
                                          concept_names=CONCEPT_UNIQUE_NAMES)

    return df_tcav.merge(df_attributes, on='path', how='left')


def normalize_tcav_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Accept both the space-separated and underscore column spellings.

    Earlier revisions wrote `"true label"` / `"concept name"` / `"cav acc"` from
    one script and `true_label` / `concept_name` from another, so the analysis
    notebooks each only worked with one of the two CSV flavours. Everything is
    written with underscores now; this keeps the older files readable.
    """
    renames = {c: c.replace(" ", "_") for c in df.columns if " " in c}
    return df.rename(columns=renames) if renames else df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=sorted(DATASET_LABELS),
                        help="which label space the trained model uses")
    parser.add_argument("--attributes", type=Path, required=True,
                        help="probability-vector CSV produced by prob_vector.py")
    parser.add_argument("--model", type=Path, required=True, help="trained .pt classifier")
    parser.add_argument("--out", type=Path, required=True, help="destination CSV")
    parser.add_argument("--model-id", default=None,
                        help="CAV cache namespace (default: <dataset>-<model file stem>)")
    parser.add_argument("--limit", type=int, default=None,
                        help="debug only: score just the first N samples")
    parser.add_argument("--layer", default=TCAV_LAYER,
                        help=f"convolution layer the CAVs are fitted in (default: {TCAV_LAYER})")
    parser.add_argument("--no-random-in-set", action="store_true",
                        help="multiclass only: fit over the 12 concepts without the random "
                             "baseline; the grid-search winner for reproducing Figure 15")
    parser.add_argument("--multiclass-cavs", action="store_true",
                        help="fit one classifier over all 12 concepts + random instead of 12 "
                             "pairwise concept-vs-random classifiers; the only design measured "
                             "to give a concept space of rank > 1 (see REPRODUCING.md)")
    parser.add_argument("--legacy-captum-classifier", action="store_true",
                        help="use captum's DefaultClassifier, whose reported accuracy is the "
                             "test-split class balance; combine with --raw-concept-scale to "
                             "reproduce the configuration the book's figures used")
    parser.add_argument("--raw-concept-scale", action="store_true",
                        help="feed concept patches as [0,1] instead of dB; reproduces the "
                             "original behaviour, which fits CAVs to chance-level directions")
    args = parser.parse_args()

    scale_to_db = not args.raw_concept_scale
    model_id = args.model_id or f"{args.dataset}-{args.model.stem}"
    if not scale_to_db:
        model_id += "-rawscale"          # keep the CAV caches apart
    if args.legacy_captum_classifier:
        model_id += "-legacyclf"
    if args.multiclass_cavs:
        model_id += "-multiclass"       # a multiclass CAV must not reuse pairwise caches
    if args.no_random_in_set:
        model_id += "-norand"
    if args.layer != TCAV_LAYER:
        model_id += "-" + args.layer.replace(".", "_")

    df_merged = get_tcav_per_sample(
        attribute_csv_path=args.attributes,
        model_path=args.model,
        label_2_index=label_to_index(DATASET_LABELS[args.dataset]),
        model_id=model_id,
        limit=args.limit,
        scale_to_db=scale_to_db,
        legacy_classifier=args.legacy_captum_classifier,
        multiclass=args.multiclass_cavs,
        include_random=not args.no_random_in_set,
        layer=args.layer,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(args.out, index=False)
    print(f"wrote {len(df_merged)} rows to {args.out}")
