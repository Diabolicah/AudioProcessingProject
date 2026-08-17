"""A concept classifier that reports a truthful CAV accuracy.

Why this exists
---------------
Captum's `DefaultClassifier` wraps a scikit-learn `SGDClassifier` in a linear
torch module. For a **binary** problem - which is exactly what TCAV's
concept-vs-random setup is - sklearn gives `coef_` shape `(1, n_features)`, so
the wrapped module emits one column per sample. `train_and_eval` then does

    predict = self.lm(x_test)                       # shape (N, 1)
    predict = self.lm.classes()[argmax(predict, 1)]  # argmax over 1 column -> 0

which always predicts the first class. The reported `accs` is therefore just the
proportion of positive examples that happened to land in the random test split,
not an accuracy at all.

Measured on perfectly separable data (60 positives, 100 randoms), captum 0.9.0
returns 0.25 where the true accuracy is 1.00; the value tracks the 0.375 class
balance, not the model.

This matters for the project book: section 5.1.5 keeps only CAVs at "at least
85% confidence", and Tables 5, 13 and 16 ("Good Cavs") are built from that
filter. Run on the default classifier, that filter selects on class balance
noise rather than on concept separability.

`BinaryConceptClassifier` below implements the same `Classifier` interface with
a real train/test split and a real accuracy, and returns weights in the shape
TCAV expects.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from captum.concept._utils.classifier import Classifier
from sklearn.linear_model import SGDClassifier
from torch import Tensor
from torch.utils.data import DataLoader

DEFAULT_TEST_SPLIT_RATIO = 0.33
DEFAULT_SEED = 42


class BinaryConceptClassifier(Classifier):
    """Linear concept classifier with a correctly computed accuracy.

    :param alpha: L2 penalty of the underlying SGDClassifier (captum's default
        is 0.01; kept so CAV directions stay comparable).
    :param seed: seed for the stratified shuffle of the train/test split.
    """

    def __init__(self, alpha: float = 0.01, max_iter: int = 1000, tol: float = 1e-3,
                 seed: int = DEFAULT_SEED) -> None:
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.lm: Optional[SGDClassifier] = None
        self._classes: List[int] = []

    def train_and_eval(self, dataloader: DataLoader,
                       test_split_ratio: float = DEFAULT_TEST_SPLIT_RATIO,
                       **kwargs: Any) -> Union[Dict, None]:
        from sklearn.model_selection import train_test_split

        inputs, labels = [], []
        for batch_inputs, batch_labels in dataloader:
            inputs.append(batch_inputs)
            labels.append(batch_labels)

        x = torch.cat(inputs).detach().cpu().numpy()
        y = torch.cat(labels).detach().cpu().numpy().astype(int)
        x = x.reshape(x.shape[0], -1)

        # Stratify so every concept is represented in train and test even when
        # the concept and random sets have different sizes.
        counts = np.bincount(y[y >= 0]) if y.min() >= 0 else None
        stratify = y if len(np.unique(y)) > 1 and (counts is None or counts[counts > 0].min() >= 2) else None
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_split_ratio, random_state=self.seed, stratify=stratify)

        self.lm = SGDClassifier(alpha=self.alpha, max_iter=self.max_iter,
                                tol=self.tol, random_state=self.seed)
        self.lm.fit(x_train, y_train)
        self._classes = [int(c) for c in self.lm.classes_]

        pred = self.lm.predict(x_test)
        accuracy = float((pred == y_test).mean())
        # Majority-class rate, so a "good" CAV can be told apart from a CAV that
        # merely predicts whichever set is larger.
        majority = float(np.bincount(y_test).max() / len(y_test))

        stats: Dict[str, Any] = {"accs": accuracy, "majority_baseline": majority}
        if len(self._classes) > 2:
            # Multi-concept experimental set: captum stores a single stats dict
            # per CAV, so expose per-concept recall for the "good CAVs" filter
            # (with 12 mutually-confusable concepts the overall accuracy alone
            # cannot say which concepts are reliable).
            stats["per_class_accs"] = torch.tensor([
                float((pred[y_test == c] == c).mean()) if (y_test == c).any() else float("nan")
                for c in self._classes
            ])
        return stats

    def weights(self) -> Tensor:
        """Concept directions, one row per concept in `classes()`.

        `TCAV._tcav_sub_computation` indexes the CAV rows by concept, so a
        binary problem must return two rows even though sklearn's `coef_` has
        one. Captum's own classifier stacks `[-coef, coef]`; the same convention
        is used here so the sign of `magnitude` stays comparable with results
        produced by the default classifier.
        """
        if self.lm is None:
            raise RuntimeError("train_and_eval must be called before weights()")
        weights = torch.tensor(np.asarray(self.lm.coef_), dtype=torch.float32)
        if weights.shape[0] == 1:
            weights = torch.stack([-1 * weights[0], weights[0]])
        return weights

    def classes(self) -> List[int]:
        """The label values captum passed in - these are *concept ids*, not 0/1."""
        return self._classes


class TorchLinearConceptClassifier(Classifier):
    """Multiclass linear concept classifier fitted on the GPU.

    Full-batch cross-entropy with Adam and L2 weight decay. This is the
    configuration a grid search over (layer x loss x weight decay x set design)
    found to reproduce the book's Figure 15 PCA spectrum: at
    `module3.blocks.0.conv2`, weight decay 1e-3, concepts-only multiclass set,
    the per-sample concept space has explained variance 78.7 / 11.0 against the
    book's 79.8 / 14.6 (see REPRODUCING.md, section 6).

    Deterministic for a fixed seed; runs on CUDA when available.
    """

    def __init__(self, weight_decay: float = 2e-3, steps: int = 400,
                 lr: float = 0.05, seed: int = DEFAULT_SEED) -> None:
        self.weight_decay = weight_decay
        self.steps = steps
        self.lr = lr
        self.seed = seed
        self._weights: Optional[Tensor] = None
        self._classes: List[int] = []

    def train_and_eval(self, dataloader: DataLoader,
                       test_split_ratio: float = DEFAULT_TEST_SPLIT_RATIO,
                       **kwargs: Any) -> Union[Dict, None]:
        import torch.nn.functional as F

        inputs, labels = [], []
        for batch_inputs, batch_labels in dataloader:
            inputs.append(batch_inputs)
            labels.append(batch_labels)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.cat(inputs).float().reshape(len(torch.cat(labels)), -1).to(device)
        raw_y = torch.cat(labels).long()

        self._classes = sorted(int(c) for c in raw_y.unique())
        remap = {c: i for i, c in enumerate(self._classes)}
        y = torch.tensor([remap[int(v)] for v in raw_y], device=device)
        n_classes = len(self._classes)

        torch.manual_seed(self.seed)
        generator = torch.Generator().manual_seed(self.seed)
        n_test = max(1, int(len(y) * test_split_ratio))
        perm = torch.randperm(len(y), generator=generator).to(device)
        test_idx, train_idx = perm[:n_test], perm[n_test:]

        def fit(idx):
            W = torch.zeros(n_classes, x.shape[1], device=device, requires_grad=True)
            b = torch.zeros(n_classes, device=device, requires_grad=True)
            opt = torch.optim.Adam([W, b], lr=self.lr, weight_decay=self.weight_decay)
            for _ in range(self.steps):
                opt.zero_grad()
                loss = F.cross_entropy(x[idx] @ W.T + b, y[idx])
                loss.backward()
                opt.step()
            return W.detach(), b.detach()

        # Holdout fit gives the honest accuracy...
        W_holdout, b_holdout = fit(train_idx)
        with torch.no_grad():
            pred = (x[test_idx] @ W_holdout.T + b_holdout).argmax(1)
            y_test = y[test_idx]
            accuracy = float((pred == y_test).float().mean())
            per_class = torch.tensor([
                float((pred[y_test == i] == i).float().mean()) if (y_test == i).any() else float("nan")
                for i in range(n_classes)
            ])
            majority = float(torch.bincount(y_test).max().item() / len(y_test))

        # ...the CAV itself is refitted on all patches, matching the fit the
        # grid search validated against Figure 15.
        W_full, _ = fit(torch.arange(len(y), device=device))
        self._weights = W_full.cpu()
        stats: Dict[str, Any] = {"accs": accuracy, "majority_baseline": majority}
        if n_classes > 2:
            stats["per_class_accs"] = per_class
        return stats

    def weights(self) -> Tensor:
        if self._weights is None:
            raise RuntimeError("train_and_eval must be called before weights()")
        w = self._weights
        if w.shape[0] == 1:
            w = torch.stack([-1 * w[0], w[0]])
        return w

    def classes(self) -> List[int]:
        return self._classes
