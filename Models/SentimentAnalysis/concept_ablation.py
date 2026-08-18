"""Concept insertion / deletion faithfulness curves for the TCAV concepts.

Insertion-deletion (Petsiuk et al., RISE) asks whether an explanation is
*faithful*: if you remove what the explanation calls important, the model's
confidence in its own prediction should fall quickly; if you add it back to an
uninformative baseline, confidence should rise quickly. Applied to pixels this
means masking regions. Applied to concepts it means intervening on the CAV
directions themselves, in the activation space where the CAVs were fitted.

For a sample whose predicted class is c, with activation `a` at the TCAV layer
and orthonormalised concept directions q_1..q_K:

    deletion(k)    a - sum_{i in top-k} (a . q_i) q_i           -> p_c should fall
    insertion(k)   a_stripped + sum_{i in top-k} (a . q_i) q_i  -> p_c should rise

where `a_stripped` is `a` with all K concept components removed, so
deletion(0) == insertion(K) == the untouched activation. The area under each
curve summarises it: a faithful ranking gives LOW deletion AUC and HIGH
insertion AUC.

Three orderings are compared, which is where the evidence actually comes from:

  tcav    concepts ranked by |TCAV magnitude| for that sample (the explanation)
  random  the same concept directions in shuffled order - does the *ranking*
          carry information, or only the subspace?
  noise   random orthonormal directions instead of CAVs, same count and same
          procedure - does the *subspace* carry information, or would any
          K-dimensional subspace do?

If `tcav` is no better than `random`, the per-concept importance ordering is not
meaningful. If `tcav` is no better than `noise`, the concepts themselves are not.

    python concept_ablation.py --tcav ravdess_tcav.csv --model <model.pt> \
        --cav "cav/<model_id>/<file>.pkl" --layer module3.blocks.1.conv2 \
        --out-dir results/ablation
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from Preprocess import audio_to_mel_spectrogram
from PreprocessParams import RAVDESS_LABELS, label_to_index

DEFAULT_SEED = 42


def load_cav_directions(cav_path: Path, n_concepts: int) -> torch.Tensor:
    """Concept directions from a captum CAV cache file, as (n_concepts, features)."""
    obj = torch.load(cav_path, weights_only=False, map_location="cpu")
    weights = obj["stats"]["weights"] if isinstance(obj, dict) and "stats" in obj else obj
    weights = torch.as_tensor(weights, dtype=torch.float32)
    if weights.shape[0] < n_concepts:
        raise ValueError(f"{cav_path} holds {weights.shape[0]} rows, need {n_concepts}")
    return weights[:n_concepts]


def orthonormalise(directions: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis spanning the same subspace.

    Without this, removing "the top 3 concepts" would double-count the shared
    component of correlated CAVs - and these CAVs are correlated - so the
    deletion curve would not correspond to removing a clean subspace.
    """
    q, _ = torch.linalg.qr(directions.T)      # (features, k)
    return q.T.contiguous()                   # (k, features)


def get_layer(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    module = model
    for part in layer_name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


class ConceptIntervention:
    """Forward hook that projects concept components out of / into an activation."""

    def __init__(self, model, layer_name: str, basis: torch.Tensor):
        self.layer = get_layer(model, layer_name)
        self.basis = basis                          # (k, features), orthonormal rows
        self.keep: Optional[torch.Tensor] = None    # (B, k) 0/1 mask
        self.mode = "off"                           # 'off' | 'delete' | 'insert'
        self._handle = self.layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self.mode == "off" or self.keep is None:
            return output
        shape = output.shape
        flat = output.flatten(1)                            # (B, features)
        coeffs = flat @ self.basis.T                        # (B, k)
        selected = coeffs * self.keep.to(flat.dtype)
        component = selected @ self.basis                   # (B, features)
        if self.mode == "delete":
            flat = flat - component
        else:                                               # insert
            flat = flat - (coeffs @ self.basis) + component
        return flat.view(shape)

    def close(self):
        self._handle.remove()


def _score(model, specs: torch.Tensor, targets: torch.Tensor,
           metric: str = "margin") -> np.ndarray:
    """Confidence in the predicted class, under the current hook state.

    `prob` is the softmax probability. It saturates: this classifier sits at
    ~0.99 on its own predictions, so a real change in the logits shows up as a
    fourth-decimal wobble and the insertion/deletion curves look flat whatever
    is done to them.

    `margin` is the predicted-class logit minus the largest competing logit -
    the same quantity monotonically, but unsaturated, so it can actually
    register the effect of removing a concept direction.
    """
    with torch.no_grad():
        logits = model(specs)
    if metric == "prob":
        return torch.softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1).cpu().numpy()

    target_logit = logits.gather(1, targets[:, None]).squeeze(1)
    competitor = logits.scatter(1, targets[:, None], float("-inf")).max(dim=1).values
    return (target_logit - competitor).cpu().numpy()


def ablation_curves(model, intervention: ConceptIntervention, specs: torch.Tensor,
                    targets: torch.Tensor, order: np.ndarray,
                    batch_size: int = 64, metric: str = "margin") -> dict:
    """Deletion and insertion curves averaged over samples.

    `order[s, j]` is the index of the j-th most important concept for sample s.
    """
    n_samples, k = order.shape
    order_t = torch.as_tensor(order, device=specs.device)

    deletion, insertion = [], []
    for step in range(k + 1):
        keep = torch.zeros(n_samples, k, device=specs.device)
        if step > 0:
            keep.scatter_(1, order_t[:, :step], 1.0)

        for mode, store in (("delete", deletion), ("insert", insertion)):
            intervention.mode = mode
            vals = []
            for i in range(0, n_samples, batch_size):
                intervention.keep = keep[i:i + batch_size]
                vals.append(_score(model, specs[i:i + batch_size],
                                   targets[i:i + batch_size], metric))
            store.append(float(np.concatenate(vals).mean()))

    intervention.mode = "off"
    deletion, insertion = np.array(deletion), np.array(insertion)
    return {
        "deletion": deletion,
        "insertion": insertion,
        "deletion_auc": float(np.trapezoid(deletion, dx=1.0 / k)),
        "insertion_auc": float(np.trapezoid(insertion, dx=1.0 / k)),
        # drop from the untouched activation to full removal - the headline
        # number: how much confidence do these concepts actually carry?
        "deletion_drop": float(deletion[0] - deletion[-1]),
    }


def build_orders(magnitudes: np.ndarray, seed: int = DEFAULT_SEED) -> dict:
    """Per-sample concept orderings: by TCAV importance, and shuffled."""
    rng = np.random.default_rng(seed)
    return {
        "tcav": np.argsort(-np.abs(magnitudes), axis=1),
        "random": np.stack([rng.permutation(magnitudes.shape[1])
                            for _ in range(magnitudes.shape[0])]),
    }


def random_basis(n_directions: int, n_features: int, seed: int = DEFAULT_SEED,
                 device="cpu") -> torch.Tensor:
    """Orthonormal directions drawn at random - the 'noise' control subspace."""
    g = torch.Generator().manual_seed(seed)
    return orthonormalise(torch.randn(n_directions, n_features, generator=g)).to(device)


def load_samples(tcav_csv: Path, concept_names: list, limit: Optional[int],
                 device, seed: int = DEFAULT_SEED):
    """Spectrograms, predicted-class indices and per-sample concept magnitudes."""
    from tcav_clustering import concept_vectors, load_tcav_csv

    raw = load_tcav_csv(tcav_csv)
    vectors = concept_vectors(raw, correct_only=True)          # correct predictions only
    if limit is not None and limit < len(vectors):
        vectors = vectors.sample(n=limit, random_state=seed).reset_index(drop=True)

    predicted = (raw[["path", "predicted_label"]].drop_duplicates()
                 .set_index("path")["predicted_label"])
    l2i = label_to_index(RAVDESS_LABELS)

    specs, targets = [], []
    for path in vectors["path"]:
        specs.append(torch.tensor(audio_to_mel_spectrogram(Path(path)), dtype=torch.float32))
        targets.append(l2i[predicted[path]])

    specs = torch.stack(specs).unsqueeze(1).to(device)
    targets = torch.tensor(targets, device=device)
    magnitudes = vectors[list(concept_names)].to_numpy(dtype=np.float32)
    return specs, targets, magnitudes, vectors


def plot_curves(results: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"tcav": "#2a7", "random": "#c80", "noise": "#a33"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, res in results.items():
        k = np.arange(len(res["deletion"]))
        axes[0].plot(k, res["deletion"], marker="o", color=colours.get(name),
                     label=f"{name} (AUC {res['deletion_auc']:.3f})")
        axes[1].plot(k, res["insertion"], marker="o", color=colours.get(name),
                     label=f"{name} (AUC {res['insertion_auc']:.3f})")

    axes[0].set_title("Deletion - remove concepts most-important first\n(lower curve = more faithful)")
    axes[1].set_title("Insertion - add concepts back, most-important first\n(higher curve = more faithful)")
    for ax in axes:
        ax.set_xlabel("number of concept directions intervened on")
        ax.set_ylabel("mean confidence in the predicted class")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    from concept_defs import CONCEPT_UNIQUE_NAMES

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tcav", type=Path, required=True, help="per-sample TCAV CSV")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cav", type=Path, required=True, help="captum CAV cache .pkl")
    parser.add_argument("--layer", default="module3.blocks.0.conv2")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=300,
                        help="samples to evaluate (default 300)")
    parser.add_argument("--metric", choices=["margin", "prob"], default="margin",
                        help="margin (predicted logit minus best competitor, unsaturated) "
                             "or prob (softmax, saturates near 1.0)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model, map_location=device, weights_only=False).to(device).eval()

    specs, targets, magnitudes, vectors = load_samples(
        args.tcav, CONCEPT_UNIQUE_NAMES, args.limit, device, args.seed)
    print(f"{len(specs)} correctly-classified samples, {magnitudes.shape[1]} concepts")

    cavs = load_cav_directions(args.cav, len(CONCEPT_UNIQUE_NAMES))
    concept_basis = orthonormalise(cavs).to(device)
    n_features = concept_basis.shape[1]
    print(f"activation space {n_features} dims; intervening on a "
          f"{concept_basis.shape[0]}-dim subspace")

    orders = build_orders(magnitudes, args.seed)
    results = {}

    for name, order in orders.items():
        iv = ConceptIntervention(model, args.layer, concept_basis)
        results[name] = ablation_curves(model, iv, specs, targets, order, metric=args.metric)
        iv.close()
        print(f"  {name:7s} deletion AUC {results[name]['deletion_auc']:.4f}   "
              f"insertion AUC {results[name]['insertion_auc']:.4f}   "
              f"drop {results[name]['deletion_drop']:+.4f}")

    noise_basis = random_basis(concept_basis.shape[0], n_features, args.seed, device)
    iv = ConceptIntervention(model, args.layer, noise_basis)
    results["noise"] = ablation_curves(model, iv, specs, targets, orders["random"],
                                       metric=args.metric)
    iv.close()
    print(f"  {'noise':7s} deletion AUC {results['noise']['deletion_auc']:.4f}   "
          f"insertion AUC {results['noise']['insertion_auc']:.4f}   "
          f"drop {results['noise']['deletion_drop']:+.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({f"{n}_{c}": results[n][c] for n in results for c in ("deletion", "insertion")}
                 ).to_csv(args.out_dir / "ablation_curves.csv", index_label="k")
    pd.DataFrame([{"ordering": n,
                   "deletion_auc": round(r["deletion_auc"], 4),
                   "insertion_auc": round(r["insertion_auc"], 4),
                   "deletion_drop": round(r["deletion_drop"], 4)}
                  for n, r in results.items()]).to_csv(args.out_dir / "ablation_auc.csv", index=False)
    plot_curves(results, args.out_dir / "ablation_curves.png")
    print(f"wrote {args.out_dir}/ablation_curves.csv, ablation_auc.csv, ablation_curves.png")


if __name__ == "__main__":
    main()
