"""
Based on the official description, the correct format for filenames in the CREMA-D dataset is:
<ActorID>_<Sentencetype>_<Emotiontype>_<EmotionIntensity>.wav

"""
from pathlib import Path 
import os
import pandas as pd

ALL_CREMA_D_DATA = Path(os.path.join("crema-d-mirror"))
# 🔁 GLOBAL: Full path to the directory containing all CREMA-D wav files
AUDIO_RELATIVE_PATH = 'AudioWAV'
AUDIO_DATA = Path(os.path.join(ALL_CREMA_D_DATA,AUDIO_RELATIVE_PATH))



emotion_map = {
    "ANG": "Anger",
    "DIS": "Disgust",
    "FEA": "Fear",
    "HAP": "Happy",
    "NEU": "Neutral",
    "SAD": "Sad"
}

intensity_map = {
    "LO": "Low",
    "MD": "Medium",
    "HI": "High",
    "XX": "Unspecified"
}

def get_actor_id(filename: str):
    if filename.endswith(".wav"):
        parts = filename.replace(".wav", "").split("_")
        if len(parts) != 4:
            return   # skip malformed filenames
        
        actor_id = int(parts[0])
        return actor_id
    
def get_sentence_type(filename: str):
    if filename.endswith(".wav"):
        parts = filename.replace(".wav", "").split("_")
        if len(parts) != 4:
            return   # skip malformed filenames
        
        sentence_type = parts[1]
        return sentence_type


def get_emotion(filename: str):
    if filename.endswith(".wav"):
        parts = filename.replace(".wav", "").split("_")
        if len(parts) != 4:
            return   # skip malformed filenames
        
        emotion = emotion_map.get(parts[2], parts[2])
        return emotion

def get_emotion_intensity(filename: str):
    if filename.endswith(".wav"):
        parts = filename.replace(".wav", "").split("_")
        if len(parts) != 4:
            return   # skip malformed filenames
        
        intensity = intensity_map.get(parts[3], parts[3])
        return intensity

def extract_crema_d_metadata_to_df(audio_dir: str, save_path: str = None) -> pd.DataFrame:
    """
    Extracts metadata from CREMA-D .wav filenames.

    Parameters:
        audio_dir (str): Path to the directory containing .wav files.
        save_path (str, optional): Path to save the output CSV. If None, doesn't save.

    Returns:
        pd.DataFrame: A DataFrame containing metadata for each file.
    """
    

    metadata = []

    for filename in os.listdir(audio_dir):
        if filename.endswith(".wav"):
            parts = filename.replace(".wav", "").split("_")
            if len(parts) != 4:
                continue  # skip malformed filenames

            actor_id = int(parts[0])
            sentence_type = parts[1]
            emotion = emotion_map.get(parts[2], parts[2])
            intensity = intensity_map.get(parts[3], parts[3])

            metadata.append({
                "filename": filename,
                "actor_id": actor_id,
                "sentence_type": sentence_type,
                "emotion": emotion,
                "emotion_intensity": intensity
            })

    df_meta = pd.DataFrame(metadata)

    if save_path:
        df_meta.to_csv(save_path, index=False)

    return df_meta
