"""Regression tests for the numbers the project book reports.

These do not need the datasets or a GPU. They pin the configuration and shapes
that the reported results depend on, so a future edit that silently changes the
preprocessing, the label space or the model geometry fails here instead of
producing quietly different tables.

    cd Models/SentimentAnalysis
    python -m pytest tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import PreprocessParams as pp
from concept_defs import CONCEPT_SPECS, CONCEPT_UNIQUE_NAMES, dir_name


# ---------------------------------------------------------------------------
# Section 3.2 - pre-processing
# ---------------------------------------------------------------------------

def test_preprocessing_matches_the_book():
    assert pp.SAMPLE_RATE == 16_000
    assert pp.N_FFT == 512
    assert pp.WINDOW_LENGTH == pp.N_FFT
    assert pp.HOP_LENGTH == pp.N_FFT // 2 == 256
    assert pp.FREQUENCY_BIN_COUNT == 64
    assert pp.MAX_SPECTOGRAM_DURATION_IN_SECONDS == 4.5


def test_target_frame_count():
    # (4.5 s * 16000 Hz - 512) // 256 + 1
    assert pp.MAX_SAMPLES == 72_000
    assert pp.TARGET_FRAMES == 280


def test_downsampled_size_is_repeated_halving_not_integer_division():
    # Three stride-2 3x3 convs with padding 1: n -> ceil(n / 2) each time.
    assert pp.downsampled_size(280) == 35
    assert pp.downsampled_size(64) == 8
    # The old `n // 8` shortcut disagrees as soon as n is not a multiple of 8.
    assert pp.downsampled_size(279) == 35 != 279 // 8


# ---------------------------------------------------------------------------
# Section 4.1 / Table 1 - label spaces
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("labels, expected_count", [
    (pp.RAVDESS_LABELS, 8),
    (pp.TESS_LABELS, 7),
    (pp.CREMAD_LABELS, 6),
])
def test_class_counts_match_table_1(labels, expected_count):
    assert len(labels) == expected_count
    assert labels == sorted(labels), "label lists must stay sorted: model indices follow sorted order"


def test_ravdess_keeps_calm_and_neutral_apart():
    assert pp.LABEL_STRINGS.CALM in pp.RAVDESS_LABELS
    assert pp.LABEL_STRINGS.NEUTRAL in pp.RAVDESS_LABELS


def test_label_index_round_trip():
    forward = pp.label_to_index(pp.RAVDESS_LABELS)
    backward = pp.index_to_label(pp.RAVDESS_LABELS)
    assert forward["angry"] == 0 and backward[0] == "angry"
    assert all(backward[idx] == label for label, idx in forward.items())


def test_disgusted_spelling_is_consistent():
    # 'disgust' vs 'disgusted' sort into the same slot for these label sets, but
    # a mismatch between modules produces two different classes downstream.
    assert pp.LABEL_STRINGS.DISGUSTED == "disgusted"
    for labels in (pp.RAVDESS_LABELS, pp.TESS_LABELS, pp.CREMAD_LABELS):
        assert "disgust" not in labels


# ---------------------------------------------------------------------------
# Section 4.4 / Figures 5-6 - the 12 concepts
# ---------------------------------------------------------------------------

def test_twelve_concepts_with_the_reported_names():
    assert len(CONCEPT_SPECS) == 12
    assert CONCEPT_UNIQUE_NAMES == sorted(set(CONCEPT_UNIQUE_NAMES))
    assert CONCEPT_UNIQUE_NAMES == [
        "long_constant_thick",
        "long_dropping_flat_thick",
        "long_dropping_steep_thick",
        "long_dropping_steep_thin",
        "long_rising_flat_thick",
        "long_rising_steep_thick",
        "long_rising_steep_thin",
        "short_constant_thick",
        "short_dropping_steep_thick",
        "short_dropping_steep_thin",
        "short_rising_steep_thick",
        "short_rising_steep_thin",
    ]


def test_concept_names_are_derived_from_the_generator():
    # Guards the bug where TCAV looked up hyphenated names while
    # create_concept_dir wrote underscored directories.
    generated = {dir_name(s["length"], s["tone"], s["rate"], s["thickness"]) for s in CONCEPT_SPECS}
    assert generated == set(CONCEPT_UNIQUE_NAMES)


def test_concept_patches_have_model_input_shape():
    pytest.importorskip("librosa", reason="concepts_creation pulls in librosa/matplotlib")
    from concepts_creation import generate_concept_patch_for_raw_input
    rng = np.random.default_rng(0)
    patch = generate_concept_patch_for_raw_input(
        mean_length=0.1, std_length=0.01, mean_tone_degree=33,
        std_tone_degree=8, mean_thickness=3.3, rng=rng)
    assert patch.shape == (pp.FREQUENCY_BIN_COUNT, pp.TARGET_FRAMES)
    assert patch.min() >= 0.0 and patch.max() <= 1.0


# ---------------------------------------------------------------------------
# Section 4.2.1 - split and hyper-parameters
# ---------------------------------------------------------------------------

def test_ravdess_split_reproduces_the_table_3_supports():
    """70-10-20 stratified split of the 1440 RAVDESS clips.

    RAVDESS records neutral at one intensity only (96 clips) and every other
    emotion at two (192 clips each). Table 3 lists test supports of
    38/38/38/39/39/19/38/39, which sum to 288 = 20% of 1440. Which three of the
    seven 192-clip classes get the extra sample is a tie-break inside
    scikit-learn, so only the multiset is asserted here.
    """
    pytest.importorskip("sklearn")
    from collections import Counter

    from sklearn.model_selection import train_test_split

    counts = {"neutral": 96, "calm": 192, "happy": 192, "sad": 192,
              "angry": 192, "fearful": 192, "disgusted": 192, "surprised": 192}
    samples = [(f"{label}_{i}", label) for label, n in counts.items() for i in range(n)]
    assert len(samples) == 1440

    train_val, test = train_test_split(
        samples, test_size=0.2, stratify=[l for _, l in samples], random_state=42)
    train, val = train_test_split(
        train_val, test_size=0.1 / 0.8, stratify=[l for _, l in train_val], random_state=42)

    assert len(test) == 288                    # 20%
    assert len(train) + len(val) + len(test) == 1440

    test_counts = Counter(l for _, l in test)
    assert test_counts["neutral"] == 19, "neutral has half the clips of every other class"
    assert sorted(test_counts.values()) == [19, 38, 38, 38, 38, 39, 39, 39]


def test_split_is_stable_across_set_iteration_order():
    """The split must not depend on how a Python set happens to iterate.

    `AllRawData` stores samples in a set; before `sorted_samples` was
    introduced, `list(that_set)` produced a different order per process (string
    hashing is randomised), so a fixed `random_state` still yielded a different
    split on every run.
    """
    pytest.importorskip("sklearn")
    pytest.importorskip("torch")
    from audio_dataset import sorted_samples

    samples = [(Path(f"/data/{i:03d}.wav"), "angry") for i in range(10)]
    shuffled = list(reversed(samples))
    assert sorted_samples(samples) == sorted_samples(shuffled)


def test_reported_hyperparameters():
    pytest.importorskip("torch")
    import main
    assert main.EPOCHS == 40
    assert main.BATCH_SIZE == 32
    assert main.LEARNING_RATE == 0.001
    assert (main.VAL_RATIO, main.TEST_RATIO) == (0.1, 0.2)


# ---------------------------------------------------------------------------
# Model geometry (Figures 1-3)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name, in_channels, num_classes", [
    ("ResNetWithAttention", 1, 8),
    ("ResNetWithAttention2d", 2, 8),
])
def test_forward_pass_shapes(model_name, in_channels, num_classes):
    torch = pytest.importorskip("torch")
    import models
    model = getattr(models, model_name)(num_classes=num_classes)
    model.eval()
    x = torch.zeros(2, in_channels, pp.FREQUENCY_BIN_COUNT, pp.TARGET_FRAMES)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, num_classes)


def test_architecture_matches_figure_1():
    pytest.importorskip("torch")
    import models
    model = models.ResNetWithAttention(num_classes=8)
    assert model.conv1.out_channels == 32 and model.conv1.kernel_size == (3, 3)
    assert model.module1.blocks[0].conv1.out_channels == 64
    assert model.module2.blocks[0].conv1.out_channels == 128
    assert model.module3.blocks[0].conv1.out_channels == 256
    assert len(model.module1.blocks) == 2, "each ResNet module holds two residual blocks (Figure 2)"
    assert model.module1.blocks[0].shortcut is True, "first block pools on the shortcut (Figure 3, left)"
    assert model.module1.blocks[1].shortcut is False, "second block has no pooling (Figure 3, right)"
    assert model.attention.embed_dim == 256 and model.attention.num_heads == 8
    assert model.fc1.out_features == 1024
    assert model.fc2.out_features == 512
    assert model.fc_out.out_features == 8


def test_tcav_layer_exists_on_the_model():
    torch = pytest.importorskip("torch")
    pytest.importorskip("captum")
    import models
    from tcav_demo import TCAV_LAYER
    model = models.ResNetWithAttention(num_classes=8)
    module = model
    for part in TCAV_LAYER.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    assert isinstance(module, torch.nn.Conv2d)


def test_gradcam_target_layer_exists():
    torch = pytest.importorskip("torch")
    import models
    model = models.ResNetWithAttention(num_classes=8)
    assert isinstance(model.module3.blocks[-1].conv2, torch.nn.Conv2d)


# ---------------------------------------------------------------------------
# CAV quality (section 5.1.5 - the "Good Cavs" filter)
# ---------------------------------------------------------------------------

def test_concept_patches_are_mapped_onto_the_model_input_range():
    """Concept patches must reach the model on the dB scale real inputs use.

    `power_to_db(..., ref=np.max)` puts real spectrograms in [-80, 0]; the
    synthetic patches are generated in [0, 1]. Feeding [0, 1] through BatchNorm
    layers calibrated for dB inputs collapses concept and random activations
    together, and the concept classifier degenerates to the majority class.
    """
    patch = np.linspace(0.0, 1.0, 100)
    scaled = pp.patch_to_db_scale(patch)
    assert scaled.min() == pytest.approx(pp.CONCEPT_DB_FLOOR)
    assert scaled.max() == pytest.approx(0.0)


def test_binary_cav_classifier_reports_a_real_accuracy():
    """Guards the captum DefaultClassifier defect.

    For a binary concept-vs-random problem captum's default always predicts the
    first class, so its reported `accs` is just the test-split class balance -
    it returns ~0.4 on perfectly separable data. The replacement must return
    1.0 there, and must not beat the majority baseline on pure noise.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("captum")
    from torch.utils.data import DataLoader, TensorDataset

    from cav_classifier import BinaryConceptClassifier

    torch.manual_seed(0)
    separable = torch.cat([torch.randn(60, 40) + 20.0, torch.randn(100, 40) - 20.0])
    # captum labels the two sets with the *concept ids*, not 0/1
    labels = torch.cat([torch.zeros(60), torch.full((100,), 12.0)]).long()

    clf = BinaryConceptClassifier()
    stats = clf.train_and_eval(DataLoader(TensorDataset(separable, labels), batch_size=16))
    assert stats["accs"] == pytest.approx(1.0)
    assert clf.classes() == [0, 12], "classes() must return captum's concept ids"
    # TCAV indexes one CAV row per concept, so a binary fit must yield two rows
    assert clf.weights().shape[0] == 2

    noise = torch.randn(160, 40)
    noise_stats = BinaryConceptClassifier().train_and_eval(
        DataLoader(TensorDataset(noise, labels), batch_size=16))
    assert noise_stats["accs"] <= noise_stats["majority_baseline"] + 0.1


def test_tcav_uses_the_fixed_classifier_and_db_scaling():
    pytest.importorskip("torch")
    pytest.importorskip("captum")
    import tcav_demo
    from cav_classifier import BinaryConceptClassifier

    assert tcav_demo.SCALE_CONCEPTS_TO_DB is True
    assert tcav_demo.BinaryConceptClassifier is BinaryConceptClassifier


# ---------------------------------------------------------------------------
# The concept dataset reproduces Figures 5 and 6
# ---------------------------------------------------------------------------

CONCEPT_GEOMETRY = {
    # concept -> (length in frames, angle in degrees) of the first patch at seed 42.
    # Measured from the generator and cross-checked against the rendered panels of
    # Figures 5-6, which show every concept drawn at ~3.5 s and ~1900 Hz because
    # each concept directory is generated from the same seed.
    "long_constant_thick":         (29, 0.7),
    "long_dropping_flat_thick":    (7, -26.6),
    "long_dropping_steep_thick":   (6, -52.7),
    "long_dropping_steep_thin":    (6, -50.9),
    "long_rising_flat_thick":      (7, 40.6),
    "long_rising_steep_thick":     (6, 59.5),
    "long_rising_steep_thin":      (6, 65.4),
    "short_constant_thick":        (12, 0.0),
    "short_dropping_steep_thick":  (4, -24.4),
    "short_dropping_steep_thin":   (3, -26.6),
    "short_rising_steep_thick":    (4, 31.0),
    "short_rising_steep_thin":     (3, 45.0),
}


def test_concept_generation_reproduces_the_book_figures():
    """The first patch of each concept, at seed 42, matches Figures 5-6.

    This pins the concept dataset itself: the book's figure panels show every
    concept as a line at ~3.5 s and ~1900 Hz with a concept-specific length and
    slope, and those are exactly what this generator produces. If this test
    fails, the concept set no longer matches the published one.
    """
    pytest.importorskip("librosa", reason="concepts_creation pulls in librosa/matplotlib")
    import numpy as np

    from concept_defs import CONCEPT_SPECS, dir_name
    from concepts_creation import generate_concept_patch_for_raw_input

    for spec in CONCEPT_SPECS:
        name = dir_name(spec["length"], spec["tone"], spec["rate"], spec["thickness"])
        rng = np.random.default_rng(42)
        # Mirror create_concept_dir exactly: it does NOT forward
        # random_thickness_boundary, so the two specs that set it were still
        # generated with the 0.5 default. Passing it here would advance the RNG
        # differently and no longer match the published figures.
        kwargs = {k: v for k, v in spec.items()
                  if k not in ("length", "tone", "rate", "thickness",
                               "random_thickness_boundary")}
        patch = generate_concept_patch_for_raw_input(rng=rng, **kwargs)

        ys, xs = np.where(patch > 0.5 * patch.max())
        expected_len, expected_angle = CONCEPT_GEOMETRY[name]

        assert int(np.ptp(xs)) + 1 == expected_len, f"{name}: line length changed"
        if np.ptp(xs) > 0:
            angle = np.degrees(np.arctan(np.polyfit(xs, ys, 1)[0]))
            assert angle == pytest.approx(expected_angle, abs=1.0), f"{name}: slope changed"

        # every panel in Figures 5-6 places the line in the last quarter of the clip
        t_centre = xs.mean() * pp.HOP_LENGTH / pp.SAMPLE_RATE
        assert 3.0 < t_centre < 4.0, f"{name}: line is not where the figures show it"
