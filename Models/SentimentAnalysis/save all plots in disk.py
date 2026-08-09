
# instantiate a ravdess dataset object to get all the paths to the ravdess dataset
from pathlib import Path
from Preprocess import audio_to_mel_spectrogram
from Visualizations import save_mel_spectrogram
from audio_dataset import RavdessRawData


ravdess_dataset = RavdessRawData()

for path, _ in ravdess_dataset.all_data:
   save_mel_spectrogram(audio_to_mel_spectrogram(path), Path(str(path).replace("RAVDESS", "RAVDESS_PLOTS")).with_suffix(".jpg")) 


