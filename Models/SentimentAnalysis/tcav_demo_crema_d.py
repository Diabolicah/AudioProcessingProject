import re
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
from captum.concept import TCAV, Concept
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ConstPaths import CremaPaths
from Preprocess import audio_to_mel_spectrogram
from PreprocessParams import TARGET_FRAMES, FREQUENCY_BIN_COUNT
from concepts_creation import generate_random_pattern_spectrogram

# os.environ["MKL_ENABLE_INSTRUCTIONS"] = "AVX2"
here = Path(__file__).resolve()
DEFAULT_CREMAD_ROOT = here.parents[0]              # …/ (two levels up)
# print(DEFAULT_CREMAD_ROOT)
sys.path.append(DEFAULT_CREMAD_ROOT)

CREMAD_ROOT = CremaPaths.WAV_DATA  # <-- set your path
device="cpu"

CONCEPT_UNIQUE_NAMES = [
                        "long_constant_thick",
                        "long_dropping_flat_thick",
                        "long_dropping_steep_thick",
                        "long_dropping_steep_thin",
                        "long_rising_flat_thick",
                        "long_rising_steep_thick",
                        "long_rising_steep_thin",
                        "short_constant_thick",
                        "short_dropping_steep_thick",
                        "short_dropping_steep_thin",
                        "short_rising_steep_thick",
                        "short_rising_steep_thin"
                        ]


LABEL_EMOTION_MAPPING = {
    0: 'angry',
    1: 'disgust',
    2: 'fearful',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
}

# optional short-code map from filename to label name
EMO_CODE_TO_NAME = {
    'ANG': 'angry',
    'DIS': 'disgust',
    'FEA': 'fearful',
    'HAP': 'happy',
    'NEU': 'neutral',
    'SAD': 'sad',
}

ALLOWED_EMOTIONS = {"ANG", "DIS", "FEA", "HAP", "NEU", "SAD"}

CREMAD_PATTERN = re.compile(
    r'^(?P<actor>\d{4})_(?P<utt>[A-Z]{3})_(?P<emo>[A-Z]{3})_(?P<intensity>[A-Z]{2})\.wav$'
)

EMO_RE = re.compile(r"_(ANG|DIS|FEA|HAP|NEU|SAD)_", re.IGNORECASE)


class PreGeneratedRandomSpectrogramDataset_CREMAD(Dataset):
    """
    PyTorch Dataset that pre-generates all random spectrogram in memory.
    """

    def __init__(self, n_samples: int, freq_count = FREQUENCY_BIN_COUNT, frames = TARGET_FRAMES, rng_seed: Optional[int] = None):
        self.n_samples = n_samples
        self.freq_count = freq_count
        self.frames = frames
        self.rng = np.random.default_rng(rng_seed)

        # Pre-generate all spectrograms in memory
        self.data = np.array([generate_random_pattern_spectrogram(freq_count, frames, rng=self.rng)
                     for _ in range(n_samples)])
        self.data = torch.tensor(self.data, dtype=torch.float32).to(device=device)

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
    PyTorch Dataset that pre-generates the dataset for a specific concept in memory.
    """

    def __init__(self, n_samples: int, concept_name: str, root_concept_dir: Path = Path("positive concepts dataset") , freq_count = FREQUENCY_BIN_COUNT, frames_count = TARGET_FRAMES, rng_seed: Optional[int] = None):
        self.n_samples = n_samples
        self.concept_name = concept_name
        self.root_concept_dir = root_concept_dir
        self.freq_count = freq_count
        self.frames = frames_count
        self.rng = np.random.default_rng(rng_seed)

        # load all .npy files from root_concept_dir/concept_name
        self.data = []
        concept_dir = self.root_concept_dir / self.concept_name
        concept_dir.mkdir(exist_ok=True, parents=True)
        for npy_file in concept_dir.glob("*.npy"):
            self.data.append(np.load(npy_file))
        self.data = np.array(self.data)
        self.data = torch.tensor(self.data, dtype=torch.float32).to(device=device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Ensure shape [1, H, W] per sample
        x = self.data[idx]
        return x.unsqueeze(0)

    @property
    def get_data(self):
        return self.data


def parse_cremad_filename(path: Path) -> Optional[Tuple[str, str, str, str]]:
    """
    Parse a CREMA-D filename and return (actor, utterance, emotion_code, intensity).
    Returns None if it doesn't match the expected pattern.
    """
    m = CREMAD_PATTERN.match(path.name)
    if not m:
        return None
    return (
        m.group('actor'),
        m.group('utt'),
        m.group('emo'),
        m.group('intensity'),
    )

def list_cremad_files(root: Path, allowed_emotions: Optional[set] = None) -> List[Path]:
    """
    Recursively list all CREMA-D wavs under `root`. Optionally filter by allowed_emotions (codes like ANG/HAP/...).
    """
    wavs = []
    for p in root.rglob('*.wav'):
        parsed = parse_cremad_filename(p)
        if not parsed:
            continue
        _, _, emo_code, _ = parsed
        if ALLOWED_EMOTIONS is None or emo_code in ALLOWED_EMOTIONS:
            wavs.append(p)
    return wavs

def group_by_emotion_cremad(paths: List[Path]) -> Dict[str, List[Path]]:
    """
    Group paths by normalized label name using `emo_code_to_name`.
    """
    buckets: Dict[str, List[Path]] = {name: [] for name in EMO_CODE_TO_NAME.values()}
    for p in paths:
        parsed = parse_cremad_filename(p)
        if not parsed:
            continue
        _, _, emo_code, _ = parsed
        if emo_code in EMO_CODE_TO_NAME:
            label_name = EMO_CODE_TO_NAME[emo_code]
            buckets[label_name].append(p)
    # Remove empties to avoid surprises
    return {k: v for k, v in buckets.items() if len(v) > 0}

def _tcav_dict_per_sample_to_df(scores_by_sample: dict, concept_names: list[str]) -> pd.DataFrame:
    # """
    # Flatten Captum TCAV results into a DataFrame with:
    # columns = ["label_name", "concept_name", "layer_name", "positive_percentage", "magnitude"]
    # """
    rows = []
    for path, exp_sets in scores_by_sample.items():
        # exp_key looks like "0-12" where 0 is the positive concept index, 12 is random/baseline
        for exp_key, layer_dict in exp_sets.items():
            try:
                pos_idx = int(str(exp_key).split("-")[0])
            except Exception:
                continue  # skip malformed keys
            if not (0 <= pos_idx < len(concept_names)):
                continue
            concept_name = concept_names[pos_idx]

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

                # Positive direction = index 0
                rows.append({
                    "path": path,
                    "concept_name": concept_name,
                    "layer_name": layer_name,
                    "positive_percentage": float(sc[0]),
                    "magnitude": float(mg[0]),
                })

    return pd.DataFrame(rows, columns=[
        "path", "concept_name", "layer_name", "positive_percentage", "magnitude"
    ])
    
def _get_tcav_dict_per_sample(all_filtered_data: pd.DataFrame): 
    
    # -----------------------------
    # 1️⃣ Load pretrained model
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(Path("ResNetWithAttention.pt"), map_location=device, weights_only=False)
    model.eval()

    # -----------------------------
    # 2️⃣ Choose layer for TCAV to work on
    # -----------------------------

    layer = "module3.blocks.0.conv2"

    # -----------------------------
    # 3️⃣ Compute TCAV
    # -----------------------------
    # Captum TCAV expects a dictionary of concept activations, with positive and negative examples.


    # Define TCAV object
    tcav = TCAV(model, [layer])


    positive_concepts: list[Concept] = [Concept(id=concept_idx, name=concept_name, data_iter=DataLoader(PreGeneratedConceptDataset(n_samples=100, concept_name=concept_name), shuffle=False))
                                for concept_idx, concept_name in enumerate(CONCEPT_UNIQUE_NAMES)]

    # This concept is the negative of concepts.
    negative_concept_dataset = PreGeneratedRandomSpectrogramDataset_CREMAD(n_samples=100, freq_count=FREQUENCY_BIN_COUNT, frames=TARGET_FRAMES)
    random_concept = Concept(id=len(positive_concepts), name='random', data_iter=DataLoader(negative_concept_dataset, shuffle=False))

    # Debug call, don't uncomment
    # show_arrays_in_separate_windows(negative_concept_dataset.get_data)

    print("Reached tcav interpret")
    
    tcav_dict_per_sample = {}
    
    # for row(pandas series) in df:
    for i, row in tqdm(all_filtered_data.iterrows(), total=len(all_filtered_data), desc="Processing samples"):
        label_name = row['predicted_label']
        path = row['path']
        sample = torch.tensor(audio_to_mel_spectrogram(Path(path)), dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape [1, 1, H, W]
        label_index = list(LABEL_EMOTION_MAPPING.keys())[list(LABEL_EMOTION_MAPPING.values()).index(label_name)]
        tcav_dict_per_sample[path] = {}
        
        score_for_label = tcav.interpret(
                inputs=sample,
                experimental_sets=[[c, random_concept] for c in positive_concepts],
                target=label_index
            )
        
        tcav_dict_per_sample[path] = score_for_label
        
    
    return tcav_dict_per_sample

def parse_cremad_emotion(p: Path):
    m = EMO_RE.search(p.name.upper())
    return EMO_CODE_TO_NAME.get(m.group(1).upper()) if m else None

def get_tcav_per_sample():
    rows = []
    for wav in CREMAD_ROOT.rglob("*.wav"):
        emo = parse_cremad_emotion(wav)
        if emo is None:
            continue
        rows.append(
            {
                "path": str(wav.resolve()),
                # filename-based ground truth for convenience
                "true_label": emo,
                # keep per-sample loop compatible: provide predicted_label
                # if you don't have a predictions table, default to true_label
                "predicted_label": emo,
                # optional; fill NaN if you don't have it
                "predicted_probability": np.nan,
            }
        )

    if not rows:
        raise ValueError(f"No CREMA-D .wav files with recognizable emotion codes under {CREMAD_ROOT}")

    df_attributes = pd.DataFrame(rows)


    dic = _get_tcav_dict_per_sample(df_attributes)
    df_tcav = _tcav_dict_per_sample_to_df(dic, CONCEPT_UNIQUE_NAMES)

    # Merge TCAV results with attributes by 'path'
    df_merged = df_tcav.merge(df_attributes, on='path', how='left')

    # Reorder columns to match your RAVDESS output shape
    desired_order = [
        'path',
        'true_label',
        'predicted_label',
        'predicted_probability',
        'concept_name',
        'layer_name',
        'positive_percentage',
        'magnitude'
    ]
    # Keep only columns that actually exist (predicted_probability may be NaN but present)
    existing = [c for c in desired_order if c in df_merged.columns]
    df_merged = df_merged[existing]

    return df_merged


if __name__ == '__main__':
    df_tcav_per_sample = get_tcav_per_sample()
    df_tcav_per_sample.to_csv("crema_d_tcav_results_per_sample.csv", index=False, encoding="utf-8")
