import itertools
from pathlib import Path

import numpy as np

from Visualizations import save_mel_spectrogram


def get_file_from_subfolders(root_folder: str | Path, n_files: int = 2):
    """
    Iterates through all subfolders in `root_folder` and yields up to `n_files` from each subfolder.

    Args:
        root_folder (str | Path): Path to the root folder.
        n_files (int): Number of files to fetch from each subfolder.

    Yields:
        tuple[str]: Paths of files collected from each subfolder.
    """
    root = Path(root_folder)

    for subfolder in root.iterdir():
        if subfolder.is_dir():
            # Collect files in this subfolder
            files = [f for f in subfolder.iterdir() if f.is_file()]

            # Take only the first `n_files`
            selected = list(itertools.islice(files, n_files))

            for file in selected:
                yield Path(file)


# Example usage:
if __name__ == "__main__":
    root_path = Path("positive concepts dataset")
    for file in get_file_from_subfolders(root_path, n_files=2):
        # load file as np array
        ndarr = np.load(file)
        # plot_mel_spectrogram(ndarr) #* debugging
        # in file, change "positive concepts dataset" to "positive concepts plots" and change suffix to .jpg
        new_plot_path = Path(str(file).replace("positive concepts dataset", "positive concepts plots")).with_suffix(".jpg")
        save_mel_spectrogram(ndarr, new_plot_path)
        print(f"Saved plot to {new_plot_path}")

