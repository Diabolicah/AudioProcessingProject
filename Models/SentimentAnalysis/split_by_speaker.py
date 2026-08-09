import random
import re
from pathlib import Path
from typing import Set, Tuple, Iterable, List

ORIGINAL_RE = re.compile(r"^\d{2}(?:-\d{2}){6}\.wav$")
AUGMENT_RE  = re.compile(r"^\d{2}(?:-\d{2}){6}_[a-zA-Z0-9]+\.wav$")

def list_actor_dirs(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("Actor_*") if p.is_dir())

def collect_files(actor_dirs: Iterable[Path], include_aug: bool) -> Set[Path]:
    paths: Set[Path] = set()
    for actor in actor_dirs:
        for wav in actor.glob("*.wav"):
            if ORIGINAL_RE.match(wav.name) or (include_aug and AUGMENT_RE.match(wav.name)):
                paths.add(wav.resolve())
    return paths

def make_actor_split(data_root: Path, n_train_actors: int = 18, seed: int = 42
                     ) -> Tuple[Set[Path], Set[Path]]:
    actors = list_actor_dirs(data_root)
    random.seed(seed); random.shuffle(actors)
    train_dirs, val_dirs = actors[:n_train_actors], actors[n_train_actors:]
    train_paths = collect_files(train_dirs, include_aug=True)
    val_paths   = collect_files(val_dirs, include_aug=False)
    return train_paths, val_paths
