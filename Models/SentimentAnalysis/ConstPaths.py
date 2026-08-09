import os.path
from enum import EnumType
from pathlib import Path

class MeldPaths(EnumType):
    AUDIO_FILES_DATA = Path(os.path.join("wav_splits"))
    TRAIN_DATA_CSV = Path(os.path.join("train.csv"))
    DEV_DATA_CSV = Path(os.path.join("dev.csv"))
    TEST_DATA_CSV = Path(os.path.join("test.csv"))

class RavdessPaths(EnumType):
    ALL_AUDIO_DATA = Path(os.path.join(r"RAVDESS\RAVDESS DATASET"))
    AUDIO_ORIGINAL_DATA = Path(os.path.join(ALL_AUDIO_DATA, 'original_data'))
    AUDIO_NEUTRAL_SYNTHESIZED_DATA = Path(os.path.join(ALL_AUDIO_DATA, 'neutral_synthesized'))
    TXT_FOR_DEEPFAKE_PATH = Path(os.path.join(ALL_AUDIO_DATA, "txt_for_deepfake"))
    #?depracated WORD_CHUNKED_AUDIO_DATA = Path(os.path.join(ALL_AUDIO_DATA, "chunked_word_audio")) #?depracated
    DOUBLE_SENTENCE_AUDIO_DATA = Path(os.path.join(ALL_AUDIO_DATA, "double_sentence_audio"))

class CremaPaths(EnumType):
    ALL_AUDIO_DATA = Path(os.path.join(r"CREMA-D\DATASETS\splitted data"))
    WAV_DATA = Path(os.path.join(ALL_AUDIO_DATA, 'AudioWAV'))
    TRAIN_DATA = Path(os.path.join(ALL_AUDIO_DATA, "train"))
    VAL_DATA = Path(os.path.join(ALL_AUDIO_DATA, "val"))
    TEST_DATA = Path(os.path.join(ALL_AUDIO_DATA, "test"))
    
class TessPaths(EnumType):
    ALL_DATA = Path(r"TESS\TESS_DATASET")
    TRAIN_DATA = Path(os.path.join(ALL_DATA, "train"))
    VAL_DATA = Path(os.path.join(ALL_DATA, "val"))
    TEST_DATA = Path(os.path.join(ALL_DATA, "test"))
    PROB_VECTOR_SHUFFLED = Path(ALL_DATA, "prob_vector_tables", "tess_spk_shuffled_prob_vector.csv")
    PROB_VECTOR_SPLITTED = Path(ALL_DATA, "prob_vector_tables", "tess_spk_splitted_prob_vector.csv")
    
class conceptPaths(EnumType):
    ALL_CONCEPTS = Path(os.path.join(r"positive concepts\positive concepts dataset"))
    