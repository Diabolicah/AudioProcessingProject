"""Dataset locations, relative to the directory the scripts are run from.

The datasets themselves are git-ignored (see .gitignore), so these paths
describe the on-disk layout the pipeline expects:

    RAVDESS/RAVDESS DATASET/
        original_data/Actor_01/03-01-01-01-01-01-01.wav ...
        neutral_synthesized/Actor_01/kids_rep1_act1.wav ...
        txt_for_deepfake/Actor_01/kids_rep1_act1.txt ...
        double_sentence_audio/Actor_01/...
    CREMA-D/DATASETS/splitted data/{train,val,test}/*.wav
    TESS/TESS_DATASET/{train,val,test}/*.wav
    positive concepts/positive concepts dataset/<concept_name>/*.npy
"""

import os.path
from pathlib import Path


class MeldPaths:
    AUDIO_FILES_DATA = Path(os.path.join("wav_splits"))
    TRAIN_DATA_CSV = Path(os.path.join("train.csv"))
    DEV_DATA_CSV = Path(os.path.join("dev.csv"))
    TEST_DATA_CSV = Path(os.path.join("test.csv"))


class RavdessPaths:
    ALL_AUDIO_DATA = Path(os.path.join(r"RAVDESS\RAVDESS DATASET"))

    # Sub-directory names *relative to* ALL_AUDIO_DATA. RavdessRawDataWithNeutral
    # joins these onto its data root, so they must stay relative.
    ORIGINAL_RELATIVE_PATH = "original_data"
    NEUTRAL_RELATIVE_PATH = "neutral_synthesized"

    AUDIO_ORIGINAL_DATA = Path(os.path.join(ALL_AUDIO_DATA, ORIGINAL_RELATIVE_PATH))
    AUDIO_NEUTRAL_SYNTHESIZED_DATA = Path(os.path.join(ALL_AUDIO_DATA, NEUTRAL_RELATIVE_PATH))
    TXT_FOR_DEEPFAKE_PATH = Path(os.path.join(ALL_AUDIO_DATA, "txt_for_deepfake"))
    DOUBLE_SENTENCE_AUDIO_DATA = Path(os.path.join(ALL_AUDIO_DATA, "double_sentence_audio"))


class CremaPaths:
    ALL_AUDIO_DATA = Path(os.path.join(r"CREMA-D\DATASETS\splitted data"))
    WAV_DATA = Path(os.path.join(ALL_AUDIO_DATA, 'AudioWAV'))
    TRAIN_DATA = Path(os.path.join(ALL_AUDIO_DATA, "train"))
    VAL_DATA = Path(os.path.join(ALL_AUDIO_DATA, "val"))
    TEST_DATA = Path(os.path.join(ALL_AUDIO_DATA, "test"))


class TessPaths:
    ALL_DATA = Path(r"TESS\TESS_DATASET")
    TRAIN_DATA = Path(os.path.join(ALL_DATA, "train"))
    VAL_DATA = Path(os.path.join(ALL_DATA, "val"))
    TEST_DATA = Path(os.path.join(ALL_DATA, "test"))
    PROB_VECTOR_SHUFFLED = Path(ALL_DATA, "prob_vector_tables", "tess_spk_shuffled_prob_vector.csv")
    PROB_VECTOR_SPLITTED = Path(ALL_DATA, "prob_vector_tables", "tess_spk_splitted_prob_vector.csv")


class conceptPaths:
    ALL_CONCEPTS = Path(os.path.join(r"positive concepts\positive concepts dataset"))
