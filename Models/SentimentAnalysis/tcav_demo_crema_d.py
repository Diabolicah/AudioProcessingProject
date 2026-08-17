"""CREMA-D filename helpers plus a filename-only TCAV shortcut.

The TCAV machinery itself lives in `tcav_demo.py`; this module used to be a
near-verbatim copy of it, which is how the two files drifted apart (one wrote
`concept_name`, the other `"concept name"`; one used underscore concept
directory names, the other hyphens). Everything shared now comes from
`tcav_demo`.

The reported CREMA-D results (Tables 15-17, 19, Figures 29-32) were produced
from *model predictions*:

    python prob_vector.py cremad --model <model.pt> --out cremad_prob_vector.csv
    python tcav_demo.py cremad --attributes cremad_prob_vector.csv \
        --model <model.pt> --out cremad_tcav.csv

`build_attribute_frame_from_filenames` below is a convenience for smoke-testing
TCAV without a probability table. It sets `predicted_label = true_label`, i.e.
it assumes a perfect classifier, so its scores are NOT the reported ones.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ConstPaths import CremaPaths
from PreprocessParams import CREMAD_LABELS, LABEL_STRINGS, label_to_index
from tcav_demo import CONCEPT_UNIQUE_NAMES, get_tcav_per_sample  # noqa: F401 - re-exported

__all__ = ["CONCEPT_UNIQUE_NAMES", "get_tcav_per_sample", "parse_cremad_filename",
           "list_cremad_files", "group_by_emotion_cremad",
           "build_attribute_frame_from_filenames"]

CREMAD_ROOT = CremaPaths.WAV_DATA

# CREMA-D filename: <ActorID>_<SentenceType>_<EmotionType>_<EmotionIntensity>.wav
EMO_CODE_TO_NAME = {
    'ANG': LABEL_STRINGS.ANGRY,
    'DIS': LABEL_STRINGS.DISGUSTED,
    'FEA': LABEL_STRINGS.FEARFUL,
    'HAP': LABEL_STRINGS.HAPPY,
    'NEU': LABEL_STRINGS.NEUTRAL,
    'SAD': LABEL_STRINGS.SAD,
}

ALLOWED_EMOTIONS = set(EMO_CODE_TO_NAME)

CREMAD_PATTERN = re.compile(
    r'^(?P<actor>\d{4})_(?P<utt>[A-Z]{3})_(?P<emo>[A-Z]{3})_(?P<intensity>[A-Z]{2})\.wav$'
)


def parse_cremad_filename(path: Path) -> Optional[Tuple[str, str, str, str]]:
    """
    Parse a CREMA-D filename and return (actor, utterance, emotion_code, intensity).
    Returns None if it doesn't match the expected pattern.
    """
    m = CREMAD_PATTERN.match(path.name)
    if not m:
        return None
    return m.group('actor'), m.group('utt'), m.group('emo'), m.group('intensity')


def list_cremad_files(root: Path, allowed_emotions: Optional[set] = None) -> List[Path]:
    """
    Recursively list all CREMA-D wavs under `root`, filtered by allowed_emotions
    (codes like ANG/HAP/...). Defaults to all six reported emotions.
    """
    allowed = ALLOWED_EMOTIONS if allowed_emotions is None else allowed_emotions
    wavs = []
    for p in root.rglob('*.wav'):
        parsed = parse_cremad_filename(p)
        if not parsed:
            continue
        _, _, emo_code, _ = parsed
        if emo_code in allowed:
            wavs.append(p)
    return sorted(wavs)


def group_by_emotion_cremad(paths: List[Path]) -> Dict[str, List[Path]]:
    """Group paths by normalised label name using EMO_CODE_TO_NAME."""
    buckets: Dict[str, List[Path]] = {name: [] for name in EMO_CODE_TO_NAME.values()}
    for p in paths:
        parsed = parse_cremad_filename(p)
        if not parsed:
            continue
        _, _, emo_code, _ = parsed
        if emo_code in EMO_CODE_TO_NAME:
            buckets[EMO_CODE_TO_NAME[emo_code]].append(p)
    return {k: v for k, v in buckets.items() if v}


def build_attribute_frame_from_filenames(root: Path = CREMAD_ROOT) -> pd.DataFrame:
    """Attribute table with `predicted_label` taken from the filename ground truth.

    Only for smoke tests - see the module docstring.
    """
    rows = []
    for wav in list_cremad_files(root):
        _, _, emo_code, _ = parse_cremad_filename(wav)
        label = EMO_CODE_TO_NAME[emo_code]
        rows.append({
            "path": str(wav.resolve()),
            "true_label": label,
            "predicted_label": label,
            "predicted_probability": np.nan,
        })

    if not rows:
        raise ValueError(f"No CREMA-D .wav files with recognizable emotion codes under {root}")

    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="trained CREMA-D .pt classifier")
    parser.add_argument("--root", type=Path, default=CREMAD_ROOT, help="CREMA-D wav root")
    parser.add_argument("--out", type=Path, default=Path("crema_d_tcav_results_per_sample.csv"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    attributes_path = args.out.with_name(args.out.stem + "_filename_attributes.csv")
    attributes_path.parent.mkdir(parents=True, exist_ok=True)
    build_attribute_frame_from_filenames(args.root).to_csv(attributes_path, index=False)

    df = get_tcav_per_sample(
        attribute_csv_path=attributes_path,
        model_path=args.model,
        label_2_index=label_to_index(CREMAD_LABELS),
        model_id=f"cremad-filenames-{args.model.stem}",
        limit=args.limit,
    )
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"wrote {len(df)} rows to {args.out}")
