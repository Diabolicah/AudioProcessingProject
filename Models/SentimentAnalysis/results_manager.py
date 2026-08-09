from __future__ import annotations

"""results_manager.py
A tiny helper that builds the directory tree for a single training **run** and
exposes convenience helpers to save artefacts (figures, JSON, checkpoints …)
under that run directory.

Designed around the folder hierarchy agreed by the team:

training_results/
    └─ {model_class}/
         └─ {raw_data_class}/
              └─ {dataset_class}/
                   └─ {run_timestamp}/
                        ├─ training_info.txt
                        ├─ accuracy_per_epoch.png
                        ├─ loss_per_epoch.png
                        └─ confusion_matrix.png

Typical usage (inside `SentimentModelHandler`):

    self.results = ResultsManager(
        model_class=self._model.__class__.__name__,
        raw_data_class=raw_data_class_name,
        dataset_class=self._train_dataset.__class__.__name__,
        user_notes=my_free_text,
    )

    # later…
    fig = _create_plot()
    self.results.save_figure(fig, "accuracy_per_epoch.png")

The module is intentionally dependency‑free (standard library only) so that it
can be imported very early and by any other module.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional
import subprocess

__all__ = ["ResultsManager"]


class ResultsManager:
    """Builds *one* run directory and keeps everything related to that run."""

    def __init__(
        self,
        *,
        model_class: str,
        raw_data_class: str,
        dataset_class: str,
        root: str | Path = "training_results",
        run_time: Optional[datetime] = None,
        user_notes: Optional[str] = None,
    ) -> None:
        
        self.model_class = model_class
        self.raw_data_class = raw_data_class
        self.dataset_class = dataset_class
        self.root = root
        self.run_time = run_time or datetime.now()
        self.user_notes = user_notes
        
        
        # Build the directory tree we want → …/training_results/Model/RawData/Dataset/2025‑06‑03_14‑57‑22
        self.base_dir: Path = (
            Path(root)
            / model_class
            / raw_data_class
            / dataset_class
            / self.run_time.strftime("%Y-%m-%d_%H-%M-%S")
        )
    
    def create_directory(self) -> None:
        """Create the run directory and write the training info file."""
        # if directory already exists, do nothing
        if self.base_dir.exists():
            return
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # A small text file with run meta‑data (and optional user notes)
        self._write_training_info(self.model_class, self.raw_data_class, self.dataset_class, self.user_notes)

    # ------------------------------------------------------------------ #
    #  Public helpers                                                    #
    # ------------------------------------------------------------------ #

    def file(self, name: str | Path) -> Path:
        """Return a *Path* object for *name* inside the run directory."""
        return self.base_dir / name

    def save_figure(self, fig, fname: str | Path) -> Path:
        """Save a Matplotlib figure and return the path written."""
        path = self.file(fname)
        fig.savefig(path, bbox_inches="tight")
        return path

    # ------------------------------------------------------------------ #
    #  Internals                                                         #
    # ------------------------------------------------------------------ #

    def _write_training_info(
        self,
        model_class: str,
        raw_data_class: str,
        dataset_class: str,
        user_notes: Optional[str],
    ) -> None:
        info = {
            "datetime": self.run_time.isoformat(timespec="seconds"),
            "model_class": model_class,
            "raw_data_class": raw_data_class,
            "dataset_class": dataset_class,
            "git_commit": self._git_commit(),
        }
        if user_notes:
            info["user_notes"] = user_notes

        with open(self.file("training_info.txt"), "w", encoding="utf-8") as fh:
            for k, v in info.items():
                fh.write(f"{k}: {v}\n")

    @staticmethod
    def _git_commit() -> str:
        """Return the current Git commit hash or 'unknown' if we are not in a repo."""
        try:
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
                .decode()
                .strip()
            )
        except Exception:
            return "unknown"
