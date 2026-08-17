from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from PreprocessParams import LABEL_STRINGS


def split_dataset_with_multi_attribute_stratification(
    original_dataset_path: Path | str,
    splitted_dataset_path: Path | str,
    attribute_fetchers: Sequence[Callable[[Path], object]],
    *,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
    allowed_extensions: Optional[Iterable[str]] = None,
    preserve_tree: bool = True,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Read files from original_dataset_path, stratify-split by multiple attributes,
    and copy them into splitted_dataset_path/{train,val,test}. Also returns and saves a manifest.

    Parameters
    ----------
    original_dataset_path : Path | str
        Root folder to scan (recursively) for files to split.
    splitted_dataset_path : Path | str
        Destination folder; will contain 'train', 'val', 'test' subfolders and manifest.csv.
    attribute_fetchers : Sequence[Callable[[Path], object]]
        Callables that receive a file Path and return an attribute value (hashable).
        The stratification label is the tuple of all returned values.
    test_size : float, default=0.2
        Fraction of files for test.
    val_size : float, default=0.1
        Fraction of *whole* (i.e., of the whole dataset) to allocate to validation.
    seed : int, default=42
        RNG seed for reproducibility.
    allowed_extensions : Iterable[str] | None, default=None
        If provided, only files with these (case-insensitive) suffixes are included (e.g., {".wav",".flac",".png"}).
    preserve_tree : bool, default=True
        If True, keep each file's relative directory structure under the split folder.
        If False, copy all files flat into the split folder.
    dry_run : bool, default=False
        If True, perform the split and return the manifest without copying files.

    Returns
    -------
    pd.DataFrame
        Manifest with columns: ['rel_path','split','attr_0','attr_1',...].
        Also saved to '<splitted_dataset_path>/manifest.csv'.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    src = Path(original_dataset_path)
    dst = Path(splitted_dataset_path)
    rng = random.Random(seed)

    if allowed_extensions:
        allowed = {e.lower() for e in allowed_extensions}
        files = [p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in allowed]
    else:
        files = [p for p in src.rglob("*") if p.is_file()]

    if not files:
        raise ValueError("No files found to split.")

    # Build attributes and labels (tuple across all attribute_fetchers)
    attrs = []
    for p in files:
        vals = []
        for f in attribute_fetchers:
            try:
                vals.append(f(p))
            except Exception as e:
                raise RuntimeError(f"Attribute fetcher {f} failed on {p}: {e}") from e
        attrs.append(tuple(vals))

    # DataFrame for convenience
    rel_paths = [p.relative_to(src).as_posix() for p in files]
    df = pd.DataFrame({"rel_path": rel_paths, "label_tuple": attrs})
    for i in range(len(attribute_fetchers)):
        df[f"attr_{i}"] = df["label_tuple"].apply(lambda t: t[i])

    # Helper: try a stratified split, with graceful fallbacks if strata are too small
    def _do_stratified(df_in: pd.DataFrame, y_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return indices for train/val/test using two-stage StratifiedShuffleSplit on df_in[y_key]."""
        y = df_in[y_key].values
        idx_all = np.arange(len(df_in))

        # Stage 1: train+val vs test
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        try:
            (trainval_idx, test_idx), = sss1.split(idx_all, y)
        except ValueError as e:
            raise ValueError(f"Stratified split (stage 1) failed on key '{y_key}': {e}")

        # Stage 2: train vs val on the train+val subset
        y_trainval = y[trainval_idx]
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size*1/(1-test_size), random_state=seed)
        try:
            (train_idx_rel, val_idx_rel), = sss2.split(trainval_idx, y_trainval)
        except ValueError as e:
            raise ValueError(f"Stratified split (stage 2) failed on key '{y_key}': {e}")

        train_idx = trainval_idx[train_idx_rel]
        val_idx = trainval_idx[val_idx_rel]
        return train_idx, val_idx, test_idx

    # Try stratifying on the full tuple; if it fails (rare/undersized strata), fall back progressively:
    # 1) full tuple, 2) first attribute, 3) random (but reproducible).
    try_order = ["label_tuple"] + (["attr_0"] if len(attribute_fetchers) > 0 else [])
    split_indices = None
    last_err = None
    for key in try_order:
        try:
            split_indices = _do_stratified(df, key)
            chosen_key = key
            break
        except ValueError as e:
            last_err = e

    if split_indices is None:
        # Fallback: random split with same proportions (no stratification)
        n = len(df)
        idx_all = list(range(n))
        rng.shuffle(idx_all)
        n_test = int(round(test_size * n))
        test_idx = np.array(idx_all[:n_test])
        trainval_idx = np.array(idx_all[n_test:])
        n_val = int(round(val_size * 1/(1-test_size) * len(trainval_idx)))
        val_idx = trainval_idx[:n_val]
        train_idx = trainval_idx[n_val:]
        split_indices = (train_idx, val_idx, test_idx)
        chosen_key = "random"
        if last_err:
            print(f"[WARN] Stratified split failed ({last_err}). Falling back to random split.")
    else:
        print(f"[INFO] Stratification key used: {chosen_key}")

    train_idx, val_idx, test_idx = split_indices

    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx,   "split"] = "val"
    df.loc[test_idx,  "split"] = "test"
    df = df.sort_values("rel_path").reset_index(drop=True)

    # Copy files
    if not dry_run:
        for split in ("train", "val", "test"):
            (dst / split).mkdir(parents=True, exist_ok=True)

        for _, row in df.iterrows():
            src_file = src / row["rel_path"]
            if preserve_tree:
                out_path = dst / row["split"] / row["rel_path"]
            else:
                # flatten
                out_path = (dst / row["split"] / Path(row["rel_path"]).name)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, out_path)

        # Save manifest
        manifest_path = dst / "manifest.csv"
        cols = ["rel_path", "split"] + [f"attr_{i}" for i in range(len(attribute_fetchers))]
        df[cols].to_csv(manifest_path, index=False)
        print(f"[INFO] Wrote manifest to {manifest_path.as_posix()}")

    return df[["rel_path", "split"] + [f"attr_{i}" for i in range(len(attribute_fetchers))]]


# Emotion label from parent directory name
def Tess_emotion_indexer(file_path: Path) -> str:
        
        file_emotion_mapping = {
            'angry': LABEL_STRINGS.ANGRY,
            'disgust': LABEL_STRINGS.DISGUSTED,
            'fear': LABEL_STRINGS.FEARFUL,
            'happy': LABEL_STRINGS.HAPPY,
            'neutral': LABEL_STRINGS.NEUTRAL,
            'ps': LABEL_STRINGS.SURPRISED,  # 'ps' stands for 'pleasant surprised'
            'sad': LABEL_STRINGS.SAD
        }
        
        # Try to find a known emotion token in the filename stem (robust to different naming conventions)
        stem = file_path.stem
        parts = stem.split('_')
        emotion_index = 2
        emotion_ds_id = parts[emotion_index]
        return file_emotion_mapping[emotion_ds_id]

def Crema_d_emotion_indexer(file_path: Path) -> str:
    index_emotion_mapping = {
            'ANG': LABEL_STRINGS.ANGRY,
            'DIS': LABEL_STRINGS.DISGUSTED,
            'FEA': LABEL_STRINGS.FEARFUL,
            'HAP': LABEL_STRINGS.HAPPY,
            'NEU': LABEL_STRINGS.NEUTRAL,
            'SAD': LABEL_STRINGS.SAD
        }

    pattern = re.compile(r'^\d{4}_[A-Z]{3}_([A-Z]{3})_[A-Z]{2}\.wav$', re.IGNORECASE)
    match = pattern.match(file_path.name)

    if not match:
        raise ValueError(f"Filename format not recognized: {file_path}")

    emotion_code = match.group(1).upper()
    emotion = index_emotion_mapping.get(emotion_code)

    if emotion is None:
        raise ValueError(f"Unknown emotion code '{emotion_code}' in file: {file_path}")

    return emotion


# Datasets that ship as one flat folder and need an on-disk 70-10-20 split
# before `CremaDSplitttedRawData` / `TessSplitttedRawData` can read them.
#
# The seed is per dataset because it is part of the published result, not a
# free choice: seed 123 on CREMA-D reproduces the exact per-class supports of
# Table 10 (train 890/890/889/889/760/890, val 127/127/127/128/109/127,
# test 254/254/255/254/218/254). Seed 42 gives the same class proportions but
# hands the one spare sample to a different class, so the supports no longer
# line up with the book. Verified against the real corpus.
SPLIT_JOBS = {
    "cremad": dict(
        original_dataset_path=r"CREMA-D\DATASETS\original data",
        splitted_dataset_path=r"CREMA-D\DATASETS\splitted data",
        attribute_fetchers=[Crema_d_emotion_indexer],
        seed=123,
    ),
    "tess": dict(
        original_dataset_path=r"TESS\TESS_ORIGINAL",
        splitted_dataset_path=r"TESS\TESS_DATASET",
        attribute_fetchers=[Tess_emotion_indexer],
        seed=42,
    ),
}


if __name__ == "__main__":
    # This block used to execute on *import*, so merely importing the module
    # copied the whole CREMA-D corpus.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(SPLIT_JOBS))
    parser.add_argument("--seed", type=int, default=None,
                        help="override the dataset's published seed (see SPLIT_JOBS)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true", help="report the split without copying")
    args = parser.parse_args()

    job = dict(SPLIT_JOBS[args.dataset])
    if args.seed is not None:
        job["seed"] = args.seed

    df_manifest = split_dataset_with_multi_attribute_stratification(
        **job,
        test_size=args.test_size,
        val_size=args.val_size,
        allowed_extensions={".wav", ".flac"},
        preserve_tree=False,
        dry_run=args.dry_run,
    )
    print(df_manifest["split"].value_counts())
    print(df_manifest.head())
