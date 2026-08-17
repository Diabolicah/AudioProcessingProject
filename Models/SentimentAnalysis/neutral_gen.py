import os
from pathlib import Path
import typing
import re
from tqdm import tqdm
# TTS (Coqui XTTS) is only needed to actually synthesise audio; it pins old
# torch/numpy versions, so it is imported lazily inside __main__ below.

from ConstPaths import RavdessPaths
from audio_dataset import RavdessRawDataWithNeutral

#: A dictionary mapping statement keys to their corresponding text phrases. # noam: check
STATEMENT_MAPPING = {
    "kids": "Kids are talking by the door",
    "dogs": "Dogs are sitting by the door",
}

def get_neutral_files_from_ravdess() -> typing.Set[Path]:
    """
    Retrieves all neutral audio files from the RAVDESS dataset.

    This function searches for .wav files with the neutral identifier ("01")
    in their filename within the `RavdessPaths.AUDIO_ORIGINAL_DATA` directory.

    :return: A set of Path objects representing the file paths of the neutral audio files.
    :rtype: typing.Set[Path]
    """
    # `*` in a glob also matches '-', so the old pattern "*-*-01-*-*-*-*.wav"
    # matched any filename containing "-01-" anywhere - e.g. the *intensity*
    # field of an angry take - and pulled non-neutral recordings in. '?' matches
    # exactly one character, which pins "01" to the emotion field.
    all_neutral_files: typing.Set[Path] = set(
        RavdessPaths.AUDIO_ORIGINAL_DATA.rglob("??-??-01-??-??-??-??.wav")
    )
    return all_neutral_files


def get_actor_number_value(file_path: Path) -> typing.Tuple[str, int]:
    """
    Extracts the actor number and value from a given file path.

    This function utilizes the `RavdessRawDataWithNeutral.get_attribute_from_filename`
    method to retrieve the actor information from the filename.

    :param file_path: The path to the audio file.
    :type file_path: Path
    :return: A tuple containing the actor number (as a string) and the actor value (as an integer).
    :rtype: typing.Tuple[str, int]
    """
    actor_number, actor_value = RavdessRawDataWithNeutral.get_attribute_from_filename(file_path, "actor")
    return actor_number, actor_value


def get_statement_number_value(file_path: Path) -> typing.Tuple[str, str]:
    """
    Extracts the statement number and value from a given file path.

    This function uses `RavdessRawDataWithNeutral.get_attribute_from_filename`
    to get the statement information.

    :param file_path: The path to the audio file.
    :type file_path: Path
    :return: A tuple containing the statement number (as a string) and the statement value (as a string).
    :rtype: typing.Tuple[str, str]
    """
    statement_number, statement_value = RavdessRawDataWithNeutral.get_attribute_from_filename(file_path, "statement")
    return statement_number, statement_value


def get_repetition_number_value(file_path: Path) -> typing.Tuple[str, int]:
    """
    Extracts the repetition number and value from a given file path.

    It leverages `RavdessRawDataWithNeutral.get_attribute_from_filename`
    to extract the repetition details.

    :param file_path: The path to the audio file.
    :type file_path: Path
    :return: A tuple containing the repetition number (as a string) and the repetition value (as an integer).
    :rtype: typing.Tuple[str, int]
    """
    repetition_number, repetition_value = RavdessRawDataWithNeutral.get_attribute_from_filename(file_path, "repetition")
    return repetition_number, repetition_value


def create_txt_files_for_deepfake() -> None:
    """
    Fetches neutral audio files and generates corresponding text files for deepfake processing.

    This function iterates through the neutral audio files in the RAVDESS dataset,
    extracts relevant information such as actor, statement, and repetition details,
    and creates a text file containing a predefined statement and the path to a
    modified neutral audio file. The modified file has the statement number swapped.
    """
    neutral_files_paths: typing.Set[Path] = get_neutral_files_from_ravdess()
    for file_path in tqdm(neutral_files_paths, "Saving txt files for deepfake_process"):
        actor_number, actor_value = get_actor_number_value(file_path)
        actor_path: Path = Path(os.path.join(RavdessPaths.TXT_FOR_DEEPFAKE_PATH, f"Actor_{actor_number}"))

        os.makedirs(actor_path, exist_ok=True)

        statement_number, statement_value = get_statement_number_value(file_path)
        repetition_number, repetition_value = get_repetition_number_value(file_path)

        saved_file_path: Path = (Path(os.path.join(actor_path, f"{statement_value}_rep{repetition_value}_act{actor_value}"))
                           .with_suffix(".txt"))

        original_name = file_path.name
        def change_fifth_number_in_file_name(filename: str, file_statement_num: str):
            """
            Changes the fifth number in a filename, swapping '01' and '02'.

            This function uses a regular expression to find the fifth number
            in the filename and swaps '01' with '02' or vice versa.

            :param filename: The name of the file to modify.
            :type filename: str
            :param file_statement_num: The statement number extracted from the filename.
            :type file_statement_num: str
            :return: The modified filename.
            :rtype: str
            """
            # Define the regex pattern to match the middle number
            pattern = r'(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2}).wav'

            # Define a function to swap '01' and '02' in the middle part
            def replace_middle(match):
                first, second, third, fourth, file_statement, sixth, seventh = match.groups()
                # Swap '01' with '02' and vice versa
                if file_statement_num == '01':
                    file_statement = '02'
                elif file_statement_num == '02':
                    file_statement = '01'
                return f"{first}-{second}-{third}-{fourth}-{file_statement}-{sixth}-{seventh}.wav"

            # Use re.sub with the replace function
            new_filename = re.sub(pattern, replace_middle, filename)
            return new_filename

        matching_neutral_file_name = change_fifth_number_in_file_name(original_name, statement_number)
        neutral_audio_path_reference: Path = file_path.with_name(matching_neutral_file_name)

        with open(saved_file_path, 'w') as saved_file:
            saved_file.writelines([STATEMENT_MAPPING.get(statement_value), '\n', neutral_audio_path_reference.__str__()])

def deepfake_and_create_synthesized_dataset(tts_model) -> None:
    r"""
    Creates a synthesized dataset for deepfake processing.

    This function generates a directory structure for the synthesized dataset,
    including subdirectories for each actor and files for each statement and repetition(actor num is also part of the filename).
    
    the directory structure is as follows:
    neutral_synthesized/
		* Actor_\<num\>
			* Exactly **four** files at the form: \$$statment$\$\_\$$repetitionNum$\$\_\$$actorNum$\$
				- spesificly: 
					* dogs_rep1_act1.wav:
							a syntesised speech saying "there are dogs..." who has been generated with the original audio of the same speeker saying "there are kids ..."
					* dogs_rep2_act1.wav
					* kids_rep1_act1.wav
					* kids_rep2_act1.wav
    the neutral_synthesized dir should be in the ALL_AUDIO_DATA folder
    """
    
    for text_file in tqdm(RavdessPaths.TXT_FOR_DEEPFAKE_PATH.rglob("*.txt"), "Creating synthesized dataset"):
        with open(text_file, 'r') as file:
            lines = file.readlines()
            statement = lines[0].strip()
            neutral_audio_path_reference = lines[1].strip()

            # Extract the actor number from the text file name
            actor_shortcut = text_file.stem.split("_")[2] # noam: need to check if stem or name is correct here
            actor_number = f'{int(actor_shortcut.split("act")[1]):02d}'
            
            actor_dir = Path(os.path.join(RavdessPaths.AUDIO_NEUTRAL_SYNTHESIZED_DATA, f"Actor_{actor_number}"))
            os.makedirs(actor_dir, exist_ok=True)
            
            # store the file path for the synthesized audio
            synthesized_audio_path = Path(os.path.join(actor_dir, text_file.stem + ".wav"))
            
            # Create the synthesized audio file (this is a placeholder, replace with actual synthesis logic)
            # Here, you would typically call your deepfake synthesis function
            # For example: synthesize_audio(statement, neutral_audio_path_reference, synthesized_audio_path)
            
            tts_model.tts_to_file(text=statement, speaker_wav=neutral_audio_path_reference, language="en", file_path=synthesized_audio_path)
            

if __name__ == '__main__':
    create_txt_files_for_deepfake()

    # import torch
    # from TTS.api import TTS
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {device}")
    # tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    #
    # deepfake_and_create_synthesized_dataset(tts)