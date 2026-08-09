import os
from pathlib import Path
from typing import Union
import re


import librosa
import soundfile as sf
import numpy as np


from ConstPaths import RavdessPaths
from PreprocessParams import SAMPLE_RATE
from neutral_gen import (get_actor_number_value,
                                                  get_statement_number_value,
                                                  get_repetition_number_value)

def get_all_ravdess_files(include_aug = False) -> set[Path]:
    ravdess_original_data_folder = RavdessPaths.AUDIO_ORIGINAL_DATA

    base_regex = r"^\d{2}(?:-\d{2}){6}"  # matches the 03-01-01-01-01-01-01 part
    if include_aug:
        pattern = re.compile(base_regex + r"(?:_[a-zA-Z0-9]+)?\.wav$")
    else:
        pattern = re.compile(base_regex + r"\.wav$")

    return {
        Path(p) for p in Path(ravdess_original_data_folder).rglob("*.wav")
        if pattern.match(p.name)
    }

def match_original_to_neutral(file_path: Path) -> Path:
    actor_number, actor_value = get_actor_number_value(file_path)
    _, statement_value = get_statement_number_value(file_path)
    repetition_number, repetition_value = get_repetition_number_value(file_path)

    matching_neutral = RavdessPaths.AUDIO_NEUTRAL_SYNTHESIZED_DATA / f"Actor_{actor_number}" / f"{statement_value}_rep{repetition_value}_act{actor_value}.wav"

    return matching_neutral


def combine_audios(file_path_original: Union[str, Path], file_path_neutral: Union[str, Path]):
    audio_path_original = Path(file_path_original)
    actor_folder, audio_file_name = audio_path_original.parts[-2:]

    audio_path_neutral = Path(file_path_neutral)

    audio_data_original, sr = librosa.load(audio_path_original, mono=True, sr=SAMPLE_RATE)
    audio_data_neutral, sr = librosa.load(audio_path_neutral, mono=True, sr=SAMPLE_RATE)

    final_audio = np.concatenate([audio_data_original, [0, 0, 0] ,audio_data_neutral])

    folder_to_create = RavdessPaths.DOUBLE_SENTENCE_AUDIO_DATA / actor_folder
    os.makedirs(folder_to_create, exist_ok=True)
    output_path = folder_to_create / audio_file_name
    print(f"Saving new file to {output_path}")

    sf.write(output_path, final_audio, samplerate=SAMPLE_RATE)

if __name__ == '__main__':
    all_ravdess_files = get_all_ravdess_files()

    for original_ravdess_file in all_ravdess_files:
        matching_file = match_original_to_neutral(original_ravdess_file)
        combine_audios(original_ravdess_file, matching_file)