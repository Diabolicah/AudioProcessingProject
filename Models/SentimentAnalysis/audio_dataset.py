import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AnyStr, Iterable, Optional, Sequence
from typing import Tuple
from collections import Counter
import os

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from ConstPaths import RavdessPaths, CremaPaths, TessPaths
from PreprocessParams import MAX_SPECTOGRAM_DURATION_IN_SECONDS, LABEL_STRINGS
from Preprocess import audio_to_mel_spectrogram, audio_to_mel_spectrogram_aligned


class AudioRawData(ABC):
    """
    Wrapper abstract class to handle saved datasets ready for pre-processing.
    """
    def __init__(self, data_root: Path, supported_formats: set[str], file_pattern: AnyStr = r".*"):
        self._data_root: Path = data_root
        self._supported_formats: set[str] = supported_formats
        self._data: set[Tuple[Path, Any]] = self._scan_supported_files(file_pattern=file_pattern) # File/s path/s, Label.  used as (Path, str) and (Dict[Path, Path], str).

        try:
            self._file_paths, self._file_labels = zip(*list(self._data))
        except ValueError:
            raise ValueError(f"Error: No data found in {data_root} with the given supported formats and file pattern.")

    def _scan_supported_files(self, file_pattern: AnyStr) -> set[Tuple[Path, Any]]:
        """
        Inner function call that saves the relevant paths of the dataset.
        :param file_pattern: Expected file pattern of the dataset for regex to handle.
        :return: A set of paths with their relevant emotion label.
        """
        base_name_pattern = re.compile(file_pattern)

        files = {
            file for file in Path(self._data_root).rglob('*')
            if file.is_file()
               and any(file.name.endswith(suffix) for suffix in self._supported_formats)
               and base_name_pattern.match(file.stem)
        }

        result = set(map(lambda x: (x, self.emotion_indexer(x)), files))
        return result

    @abstractmethod
    def emotion_indexer(self, file_path: Path) -> str:
        pass

    @property
    def all_data(self) -> set[Tuple[Path, Any]]:
        return self._data

    def print_all_label_counts(self) -> None:
        """
        Prints the count of each label in the data.
        """
        counter: dict = dict(Counter(self._file_labels))
        print(f"Total label counts: {counter}")

class RavdessRawData(AudioRawData):
    # `include_calm` defaults to True: the book (Table 1 / Table 3) reports eight
    # RAVDESS classes with `calm` and `neutral` kept apart, and the per-class
    # supports it lists (calm 38, neutral 19 in the test split) only come out of
    # the eight-class labelling. Folding calm into neutral gives seven classes
    # and different supports.
    def __init__(self, include_calm: bool = True, include_aug: bool = False):
        self._include_calm = include_calm
        self._include_aug = include_aug
        super().__init__(RavdessPaths.AUDIO_ORIGINAL_DATA, {".wav"}, r"^\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$")

    def emotion_indexer(self, file_path: Path) -> str:
        """
        For RAVDESS, label is indicated in the third number in the name. This function handles mapping it to a
        readable label.
        :param file_name: File path from RAVDESS dataset.
        :return: Label of the file according to the index.
        """
        numbers = re.findall(r'\d+', file_path.name.__str__())

        index_emotion_mapping = {
            '01': LABEL_STRINGS.NEUTRAL,
            '02': LABEL_STRINGS.NEUTRAL if not self._include_calm else LABEL_STRINGS.CALM,
            '03': LABEL_STRINGS.HAPPY,
            '04': LABEL_STRINGS.SAD,
            '05': LABEL_STRINGS.ANGRY,
            '06': LABEL_STRINGS.FEARFUL,
            '07': LABEL_STRINGS.DISGUSTED,
            '08': LABEL_STRINGS.SURPRISED
        }

        emotion_index = numbers[2]
        emotion = index_emotion_mapping[emotion_index]
        return emotion

class CREMARawData(AudioRawData):
    def __init__(self, data_root: Path = CremaPaths.ALL_AUDIO_DATA):
        super().__init__(data_root, {".wav"}, r"^\d{4}_[A-Z]{3}_[A-Z]{3}_[A-Z]{2}$")

    def emotion_indexer(self, file_path: Path) -> str:
        """
        The sentences were presented using different emotion (in parentheses is the three-letter code used in the third part of the filename):

        Anger (ANG)
        Disgust (DIS)
        Fear (FEA)
        Happy/Joy (HAP)
        Neutral (NEU)
        Sad (SAD)

        :param file_name: Name of the file to process.
        :return: String of emotion indexing.
        """
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

class TESSRawData(AudioRawData):
    def __init__(self, data_root: Path = TessPaths.ALL_DATA):
        super().__init__(data_root, {".wav"})

    def emotion_indexer(self, file_path: Path) -> str:
        
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


def sorted_samples(samples: Iterable) -> list:
    """Deterministic ordering for a set of (path, label) samples.

    The raw-data classes hold their samples in a `set`. Iteration order over a
    set of `Path` objects depends on string hashing, which Python randomises per
    process unless PYTHONHASHSEED is fixed - so `list(some_set)` produced a
    different order on every run, and `train_test_split(..., random_state=42)`
    therefore produced a *different* split on every run despite the fixed seed.
    Sorting first is what makes the reported splits reproducible.
    """
    return sorted(samples, key=lambda sample: str(sample[0]))


class AllRawData:
    def __init__(self, raw_datasets: tuple[AudioRawData, ...], val_ratio: float = 0.3):
        sets_of_data = (raw_dataset.all_data for raw_dataset in raw_datasets)
        self._all_raw_data = set().union(*sets_of_data)
        self._train_data, self._val_data, self._test_data = None, None, None

    def train_val_test_split(self, val_ratio: float = 0.1, test_ratio: float = 0.2):
        """
        Splits the data into train, validation, and test sets.
        Defaults reproduce the book's 70-10-20 split (section 4.2.1).
        :param val_ratio: Floating value between 0 and 1, ratio of data for validation.
        :param test_ratio: Floating value between 0 and 1, ratio of data for testing.
        :return: train_data, val_data, test_data
        """
        ordered_data = sorted_samples(self._all_raw_data)

        # First split off the test set
        train_val_data, self._test_data = train_test_split(
            ordered_data,
            test_size=test_ratio,
            stratify=[label for _, label in ordered_data],
            random_state=42
        )

        # Now split remaining data into train and validation
        val_adjusted_ratio = val_ratio / (1 - test_ratio)  # Adjust ratio relative to remaining data
        self._train_data, self._val_data = train_test_split(
            train_val_data,
            test_size=val_adjusted_ratio,
            stratify=[label for _, label in train_val_data],
            random_state=42
        )

        return self._train_data, self._val_data, self._test_data

    @property
    def all_data(self) -> set[tuple[Path, Any]]:
        return self._all_raw_data

    @property
    def train_data(self) -> set[tuple[Path, Any]]:
        return self._train_data

    @property
    def val_data(self) -> set[tuple[Path, Any]]:
        return self._val_data

    @property
    def test_data(self) -> set[tuple[Path, Any]]:
        return self._test_data

    def print_all_label_counts(self) -> None:
        """
        Prints all label counts for each set.
        """
        label_counts_all =  Counter([label for _, label in self._all_raw_data])
        label_counts_train = Counter([label for _, label in self.train_data])
        label_counts_val = Counter([label for _, label in self.val_data])
        print(f"Labels all: {label_counts_all}\n"
              f"Labels train: {label_counts_train}\n"
              f"Labels validation: {label_counts_val}")

    def remove_labels(self, label_list: tuple[str] = ('surprised', 'calm')) -> None:
        self._all_raw_data = set(((file_name, label) for file_name, label in self._all_raw_data if label not in label_list))


class SplitttedAudioRawData:
    def __init__(self, train_raw_data: AudioRawData, val_raw_data: AudioRawData, test_raw_data: Optional[AudioRawData], supported_formats: set[str], file_pattern: AnyStr = r".*"):
        self._train_raw_data: AudioRawData = train_raw_data
        self._val_raw_data: AudioRawData = val_raw_data
        self._test_raw_data: Optional[AudioRawData] = test_raw_data
        self._supported_formats: set[str] = supported_formats
        self._file_pattern: AnyStr = file_pattern
    
    @property
    def train_data(self) -> set[Tuple[Path, Any]]:
        return self._train_raw_data.all_data

    @property
    def val_data(self) -> set[Tuple[Path, Any]]:
        return self._val_raw_data.all_data

    @property
    def test_data(self) -> Optional[set[Tuple[Path, Any]]]:
        return self._test_raw_data.all_data if self._test_raw_data is not None else set()

    @property
    def all_data(self) -> set[Tuple[Path, Any]]:
        return (self.train_data | self.val_data | self.test_data)
    

class CremaDSplitttedRawData(SplitttedAudioRawData):
    def __init__(self):
        cremaD_train = CREMARawData(CremaPaths.TRAIN_DATA)
        cremaD_val = CREMARawData(CremaPaths.VAL_DATA)
        cremaD_test = CREMARawData(CremaPaths.TEST_DATA)
        super().__init__(cremaD_train, cremaD_val, cremaD_test, {".wav"})
        

class TessSplitttedRawData(SplitttedAudioRawData):
    def __init__(self):
        tess_train = TESSRawData(TessPaths.TRAIN_DATA)
        tess_val = TESSRawData(TessPaths.VAL_DATA)
        tess_test = TESSRawData(TessPaths.TEST_DATA)
        super().__init__(tess_train, tess_val, tess_test, {".wav"})
    
    
class EmotionSpecDataset(Dataset):
    def __init__(self, file_paths: Iterable, max_length_in_seconds: float = MAX_SPECTOGRAM_DURATION_IN_SECONDS,
                 class_names: Optional[Sequence[str]] = None, trim: bool = False):
        """
        :param class_names: Explicit label space shared by every split. Pass it
            (e.g. PreprocessParams.RAVDESS_LABELS) whenever train/val/test are
            built separately: LabelEncoder is fitted per split, so a split that
            happens to be missing a rare class would otherwise encode every
            later class to a different integer than the other splits, silently
            scrambling the labels. Defaults to the labels present in this split.
        """
        # sorted_samples keeps __getitem__ indices stable across runs even when
        # the caller hands over a set (see sorted_samples' docstring).
        self._data = sorted_samples(file_paths)
        self._paths , self._labels = zip(*self._data)

        self.__label_encoder = LabelEncoder()
        if class_names is not None:
            self.__label_encoder.fit(sorted(class_names))
            unknown = set(self._labels) - set(self.__label_encoder.classes_)
            if unknown:
                raise ValueError(f"Labels {sorted(unknown)} are not in the declared class list {sorted(class_names)}")
            self._labels = torch.tensor(self.__label_encoder.transform(self._labels))
        else:
            self._labels = torch.tensor(self.__label_encoder.fit_transform(self._labels))
        self._class_names = list(self.__label_encoder.classes_)

        # Get the number of classes
        self.num_classes = len(self.__label_encoder.classes_)

        # Compute class weights
        self.class_weights = self.__compute_class_weights()

        self.max_length_in_seconds = max_length_in_seconds
        self.trim = trim

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        file_path = self._paths[idx]
        label = self._labels[idx]

        # noam: audio_to_mel_spectrogram returns shape (freq_bins, time_frames)
        mel_spectrogram = audio_to_mel_spectrogram(file_path=file_path, max_length_in_seconds=self.max_length_in_seconds,
                                                   trim=self.trim)

        # Convert to torch.Tensor
        mel_spectrogram = torch.from_numpy(mel_spectrogram).float()
        
        # Now expand to shape (1, freq_bins, time_frames)
        mel_spectrogram = mel_spectrogram.unsqueeze(dim=0)
        
        label = label.long()

        return mel_spectrogram, label

    @property
    def class_names(self):
        return self._class_names

    @property
    def class_counts(self):
        return self._class_counts

    def decode_label(self, encoded_label):
        return self.__label_encoder.inverse_transform([encoded_label])[0]

    def __compute_class_weights(self) -> torch.Tensor:
        """
        Computes class weights based on the frequency of each class in the dataset.

        Returns:
            torch.Tensor: Tensor of class weights (inverse frequency).
        """
        class_counts = torch.bincount(self._labels, minlength=self.num_classes)
        self._class_counts = class_counts
        total_samples = len(self._labels)
        class_weights = total_samples / (class_counts + 1e-6)  # Avoid division by zero
        return class_weights.float()

class EmotionSpecDataset2d(Dataset):
    def __init__(self, data: set, class_names: Optional[Sequence[str]] = None,
                 align_durations: bool = False):
        """
        :param class_names: see :class:`EmotionSpecDataset`.
        :param align_durations: time-stretch the synthesized neutral channel to
            the duration of the original recording it is paired with.

            The depth model's hypothesis (book, section 4.2.2) is that giving
            the network a neutral rendition of the same sentence lets it
            "identify the differences between the two input". That only works if
            the two channels line up in time. They do not by default: XTTS
            speaks faster than the actors, so the synthesized channel averages
            0.67x the original's duration and 77% of pairs differ by more than
            25%, which puts a given word in different frames of the two
            channels. Stretching (phase-vocoder, pitch preserved) restores the
            frame-wise correspondence the hypothesis assumes.

            The stretch factor has to be applied per pair rather than baked into
            the synthesis, because one neutral clip is shared by every emotion
            recorded for that actor/statement/repetition.
        """
        self.align_durations = align_durations
        self._data = sorted_samples(data)
        self._paths , self._labels = zip(*self._data)

        self.__label_encoder = LabelEncoder()
        if class_names is not None:
            self.__label_encoder.fit(sorted(class_names))
            self._labels = torch.tensor(self.__label_encoder.transform(self._labels))
        else:
            self._labels = torch.tensor(self.__label_encoder.fit_transform(self._labels))
        self._class_names = list(self.__label_encoder.classes_)

        # Get the number of classes
        self.num_classes = len(self.__label_encoder.classes_)

        # Compute class weights
        self.class_weights = self.__compute_class_weights()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        file_pair = self._paths[idx]
        label = self._labels[idx]

        # noam: audio_to_mel_spectrogram returns shape (freq_bins, time_frames)
        original_mel_spectrogram = audio_to_mel_spectrogram(file_path=file_pair[0]) # original audio
        if self.align_durations:
            synthesized_mel_spectrogram = audio_to_mel_spectrogram_aligned(
                synthesized_path=file_pair[1], reference_path=file_pair[0])
        else:
            synthesized_mel_spectrogram = audio_to_mel_spectrogram(file_path=file_pair[1]) # synthesized audio
        
        # Convert to torch.Tensor
        original_mel_spectrogram = torch.from_numpy(original_mel_spectrogram).float()
        synthesized_mel_spectrogram = torch.from_numpy(synthesized_mel_spectrogram).float()
        
        # now add the two spectrograms to create a 2D tensor with shape (2, freq_bins, time_frames)
        mel_spectrogram = torch.stack((original_mel_spectrogram, synthesized_mel_spectrogram), dim=0)

        label = label.long()

        return mel_spectrogram, label

    @property
    def class_names(self):
        return self._class_names

    def decode_label(self, encoded_label):
        return self.__label_encoder.inverse_transform([encoded_label])[0]

    def __compute_class_weights(self) -> torch.Tensor:
        """
        Computes class weights based on the frequency of each class in the dataset.

        Returns:
            torch.Tensor: Tensor of class weights (inverse frequency).
        """
        class_counts = torch.bincount(self._labels, minlength=self.num_classes)
        total_samples = len(self._labels)
        class_weights = total_samples / (class_counts + 1e-6)  # Avoid division by zero
        return class_weights.float()

# add RavdessRawData in which every sample is two audio-file-paths: file2classify and originalneutral
class RavdessRawDataWithNeutral(AudioRawData):
    def emotion_indexer(self, file_name: Path) -> str:
        pass

    def __init__(self, include_calm = True, include_aug=False):
        self._include_aug = include_aug
        self._include_calm = include_calm
        super().__init__(RavdessPaths.ALL_AUDIO_DATA, {".wav"}, r"^\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$")

    def _scan_supported_files(self, file_pattern = r"^\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$") -> set[Tuple[Tuple[Path, Path], str]]:
        """
        Scans and saves the file paths of the model and a relevant label based on the index in the name.
        @return: A set of tuples, each tuple holds (file path, label).
        """
        self.neutral_data = Path(os.path.join(self._data_root, RavdessPaths.NEUTRAL_RELATIVE_PATH))
        self.original_data = Path(os.path.join(self._data_root, RavdessPaths.ORIGINAL_RELATIVE_PATH))

        base_name_pattern = re.compile(file_pattern)


        files = {
            (file, self.get_assosiated_neutral_file(file))
            for file in self.original_data.rglob('*')
            if file.is_file()
            and any(file.name.endswith(suffix) for suffix in self._supported_formats)
            and base_name_pattern.match(file.stem)
        }

        result = {
            (pair, self._label_for(pair[0]))
            for pair in files
        }

        return result

    def _label_for(self, original_file: Path) -> str:
        """Canonical emotion label, honouring the `include_calm` flag."""
        label = self.get_attribute_from_filename(original_file, "emotion")[1]
        if label == LABEL_STRINGS.CALM and not self._include_calm:
            return LABEL_STRINGS.NEUTRAL
        return label

    def __get_emotion_from_index(self, filename):
        """
        For RAVDESS, label is indicated in the third number in the name. This function handles mapping it to a
        readable label.
        @param filename: File path from RAVDESS dataset.
        @return: Label of the file according to the index.
        """
        numbers = re.findall(r'\d+', filename.name.__str__())

        index_emotion_mapping = {
            '01': 'neutral',
            '02': 'neutral' if not self._include_calm else 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': LABEL_STRINGS.DISGUSTED,
            '08': LABEL_STRINGS.SURPRISED
        }

        emotion_index = numbers[2]
        emotion = index_emotion_mapping[emotion_index]
        return emotion

    def get_assosiated_neutral_file(self, file):
        # get audio required attributes
        statement_num, statment = RavdessRawDataWithNeutral.get_attribute_from_filename(file, "statement")

        repetition_num_str, repetition_num_int = RavdessRawDataWithNeutral.get_attribute_from_filename(file, "repetition")

        actor_num_str, actor_num_int = RavdessRawDataWithNeutral.get_attribute_from_filename(file, "actor")

        # create a path to the neutral file
        neutral_file_path = Path(os.path.join(self.neutral_data, f'Actor_{actor_num_int:02d}', f"{statment}_rep{repetition_num_int}_act{actor_num_int}.wav"))

        return neutral_file_path

    @staticmethod
    def get_attribute_from_filename(filename, attribute: str):
        attribute2index = {
            "modality": 0,
            "vocal_channel": 1,
            "emotion": 2,
            "emotional_intensity": 3,
            "statement": 4,
            "repetition": 5,
            "actor": 6
        }
        num2attrvalue = {
            "modality": {"01": "full-AV", "02": "video-only", "03": "audio-only"},
            "vocal_channel": {"01": "speech", "02": "song"},
            # Spellings must match PreprocessParams.LABEL_STRINGS: the label
            # encoder sorts class names, so "disgust" and "disgusted" are two
            # different classes and mixing them across modules silently shifts
            # every model output index after 'c'.
            "emotion": {"01": LABEL_STRINGS.NEUTRAL, "02": LABEL_STRINGS.CALM, "03": LABEL_STRINGS.HAPPY,
                        "04": LABEL_STRINGS.SAD, "05": LABEL_STRINGS.ANGRY, "06": LABEL_STRINGS.FEARFUL,
                        "07": LABEL_STRINGS.DISGUSTED, "08": LABEL_STRINGS.SURPRISED},
            "emotional_intensity": {"01": "normal", "02": "strong"},
            "statement": {"01": "kids", "02": "dogs"},
            "repetition": {"01": 1, "02": 2},
            "actor": {f"{i:02d}": i for i in range(1, 25)} # maps from string number to int number
        }
        ########################################################
        # e.g. filename: "03-01-02-01-02-01-13.wav"
                # desired attribute: "actor"
        ########################################################

        # get the filename numbers |
        file_numbers = re.findall(r'\d+', filename.name.__str__()) # e.g. ['03', '01', '02', '01', '02', '01', '13']
        # get the desired attribute index
        attribute_index = attribute2index[attribute] # e.g. 6 (actor)
        # get the desired attribute number
        attribute_number = file_numbers[attribute_index] # e.g. "13"(index 6 at the filename numbers)
        # get the desired attribute value
        attribute_value = num2attrvalue[attribute][attribute_number] # e.g. "13" -> 13

        return attribute_number, attribute_value


class AudioRawDataWithOriginalNeutral(AudioRawData):
    """Superseded by RavdessRawDataWithNeutral and not used by any reported run.

    It pairs each recording with a *real* neutral take by the same actor rather
    than with the XTTS synthesis the book describes (section 4.2.2), and its
    `_scan_supported_files` override drops the base class's `file_pattern`
    argument, so instantiating it raises TypeError. Kept only for reference.
    """

    def __init__(self):
        super().__init__(RavdessPaths.AUDIO_ORIGINAL_DATA, {".wav"}) # the path that will be the data root

    def _scan_supported_files(self) -> set[Tuple[Path, str]]:
        """
        Scans and saves the file paths of the model and a relevant label based on the index in the name.
        @return: A set of tuples, each tuple holds (file path, label).
        """

        files = {
        (file, self.get_path_of_neutral(file))
        for file in self._data_root.rglob('*')
        if file.is_file() and any(file.name.endswith(suffix) for suffix in self._supported_formats)
        }

        result = set(map(lambda x: (x, AudioRawDataWithOriginalNeutral.__get_emotion_from_index(x[0])), files))
        return result

    # MICHAL
    # TODO : CHECK IF IT IS RIGHT AND LOGICAL
    def get_path_of_neutral(self , original_file : Path):

        filename = original_file.name  # e.g. "03-01-05-01-02-02-16.wav"

        # Extract numbers from filename
        numbers = re.findall(r'\d+', filename)

        # the case where the original audio is neutral - search for the same audio with different repetition value
        if self.get_attribute_from_filename(original_file,"emotion")[1] == "neutral" :
            """
            Given a neutral original RAVDESS audio file, returns the neutral version
            with different repetition value.
            """
            if self.get_attribute_from_filename(original_file,"repetition")[0] == '01' :
                # Set repetition (index 5) to "02"
                numbers[5] = '02'
            else :
                # Set repetition (index 5) to "01"
                numbers[5] = '01'

                # Reconstruct the new filename
            new_filename = '-'.join(numbers) + ".wav"

            # The new file should be in the same actor folder as the original
            actor_folder = original_file.parent  # This is .../Actor_##/

            neutral_file_path = actor_folder / new_filename
            return neutral_file_path

        # the case where the original audio is not neutral - search for the same audio in neutral and repetition =1
        else :
            """
            Given an original RAVDESS audio file, returns the corresponding neutral version
            with repetition set to 01.
            """
            # Replace emotion (index 2) with "01" for neutral
            numbers[2] = "01"

            # Set repetition (index 5) to "01"
            numbers[5] = "01"

            # Reconstruct the new filename
            new_filename = '-'.join(numbers) + ".wav"

            # The new file should be in the same actor folder as the original
            actor_folder = original_file.parent  # This is .../Actor_##/

            neutral_file_path = actor_folder / new_filename
            return neutral_file_path


    @staticmethod
    def get_attribute_from_filename(filename, attribute: str):
        attribute2index = {
            "modality": 0,
            "vocal_channel": 1,
            "emotion": 2,
            "emotional_intensity": 3,
            "statement": 4,
            "repetition": 5,
            "actor": 6
        }
        num2attrvalue = {
            "modality": {"01": "full-AV", "02": "video-only", "03": "audio-only"},
            "vocal_channel": {"01": "speech", "02": "song"},
            # Spellings must match PreprocessParams.LABEL_STRINGS: the label
            # encoder sorts class names, so "disgust" and "disgusted" are two
            # different classes and mixing them across modules silently shifts
            # every model output index after 'c'.
            "emotion": {"01": LABEL_STRINGS.NEUTRAL, "02": LABEL_STRINGS.CALM, "03": LABEL_STRINGS.HAPPY,
                        "04": LABEL_STRINGS.SAD, "05": LABEL_STRINGS.ANGRY, "06": LABEL_STRINGS.FEARFUL,
                        "07": LABEL_STRINGS.DISGUSTED, "08": LABEL_STRINGS.SURPRISED},
            "emotional_intensity": {"01": "normal", "02": "strong"},
            "statement": {"01": "kids", "02": "dogs"},
            "repetition": {"01": 1, "02": 2},
            "actor": {f"{i:02d}": i for i in range(1, 25)} # maps from string number to int number
        }
        ########################################################
        # e.g. filename: "03-01-02-01-02-01-13.wav"
                # desired attribute: "actor"
        ########################################################

        # get the filename numbers |
        file_numbers = re.findall(r'\d+', filename.name.__str__()) # e.g. ['03', '01', '02', '01', '02', '01', '13']
        # get the desired attribute index
        attribute_index = attribute2index[attribute] # e.g. 6 (actor)
        # get the desired attribute number
        attribute_number = file_numbers[attribute_index] # e.g. "13"(index 6 at the filename numbers)
        # get the desired attribute value
        attribute_value = num2attrvalue[attribute][attribute_number] # e.g. "13" -> 13

        return attribute_number, attribute_value


    @staticmethod
    def __get_emotion_from_index(filename):
        """
        For RAVDESS, label is indicated in the third number in the name. This function handles mapping it to a
        readable label.
        @param filename: File path from RAVDESS dataset.
        @return: Label of the file according to the index.
        """
        numbers = re.findall(r'\d+', filename.name.__str__())

        index_emotion_mapping = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': LABEL_STRINGS.DISGUSTED,
            '08': LABEL_STRINGS.SURPRISED
        }

        emotion_index = numbers[2]
        emotion = index_emotion_mapping[emotion_index]
        return emotion

