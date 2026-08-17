"""The numbers printed in the project book, as data.

Kept so any future run can be diffed against the published result instead of
against someone's memory of it. Every value here is transcribed from the PDF;
where the book is internally inconsistent that is noted inline.

    from reference.paper_tables import TABLE_3_RAVDESS_TEST, TABLE_6_RAVDESS_MAGNITUDE
"""

# --- Table 2: weighted averages, single-input vs depth model (RAVDESS) ------
# (precision, recall, f1)
TABLE_2_WEIGHTED = {
    "single": {"train": (1.000, 1.000, 1.000),
               "validation": (0.839, 0.832, 0.830),
               "test": (0.728, 0.722, 0.721)},
    "depth":  {"train": (1.000, 1.000, 1.000),
               "validation": (0.743, 0.734, 0.734),
               "test": (0.713, 0.710, 0.705)},
}

# --- Table 3: per-class, single-input model (RAVDESS) -----------------------
# class -> (precision, recall, f1, support)
TABLE_3_RAVDESS_VAL = {
    "angry": (0.90, 0.95, 0.93, 20), "calm": (0.94, 0.79, 0.86, 19),
    "disgusted": (0.95, 0.95, 0.95, 19), "fearful": (0.77, 0.89, 0.83, 19),
    "happy": (0.93, 0.74, 0.82, 19), "neutral": (0.67, 1.00, 0.80, 10),
    "sad": (0.63, 0.53, 0.57, 19), "surprised": (0.84, 0.89, 0.86, 18),
}
TABLE_3_RAVDESS_TEST = {
    "angry": (0.78, 0.92, 0.84, 38), "calm": (0.78, 0.76, 0.77, 38),
    "disgusted": (0.84, 0.82, 0.83, 38), "fearful": (0.68, 0.69, 0.68, 39),
    "happy": (0.72, 0.59, 0.65, 39), "neutral": (0.50, 0.74, 0.60, 19),
    "sad": (0.72, 0.61, 0.66, 38), "surprised": (0.70, 0.67, 0.68, 39),
}

# --- Tables 8/9: TESS -------------------------------------------------------
# NOTE: the book lists val surprised = 41, which makes the corpus total 2801.
# TESS has exactly 2800 clips and a stratified 70-10-20 split gives 40.
TABLE_9_TESS_MACRO = {"train": (0.988, 0.988, 0.988),
                      "validation": (0.993, 0.993, 0.993),
                      "test": (0.995, 0.995, 0.995)}
TABLE_8_TESS_SUPPORT = {"train": 280, "validation": 40, "test": 80}   # per class

# --- Tables 10/11: CREMA-D --------------------------------------------------
TABLE_11_CREMAD_MACRO = {"train": (1.000, 1.000, 1.000),
                         "validation": (0.624, 0.623, 0.622),
                         "test": (0.624, 0.618, 0.620)}
TABLE_10_CREMAD_TEST = {
    "angry": (0.784, 0.685, 0.731, 254), "disgusted": (0.566, 0.618, 0.591, 254),
    "fearful": (0.546, 0.514, 0.529, 255), "happy": (0.582, 0.642, 0.611, 254),
    "neutral": (0.681, 0.638, 0.659, 218), "sad": (0.583, 0.610, 0.596, 254),
}
TABLE_10_CREMAD_SUPPORT = {
    "train": {"angry": 890, "disgusted": 890, "fearful": 889,
              "happy": 889, "neutral": 760, "sad": 890},
    "validation": {"angry": 127, "disgusted": 127, "fearful": 127,
                   "happy": 128, "neutral": 109, "sad": 127},
    "test": {"angry": 254, "disgusted": 254, "fearful": 255,
             "happy": 254, "neutral": 218, "sad": 254},
}

# --- Concept column order used by Tables 6, 14 and 17 -----------------------
CONCEPT_ORDER = [
    "long_constant_thick", "long_dropping_flat_thick", "long_dropping_steep_thick",
    "long_dropping_steep_thin", "long_rising_flat_thick", "long_rising_steep_thick",
    "long_rising_steep_thin", "short_constant_thick", "short_dropping_steep_thick",
    "short_dropping_steep_thin", "short_rising_steep_thick", "short_rising_steep_thin",
]

# --- Table 6: average TCAV magnitude per concept per label (RAVDESS) --------
TABLE_6_RAVDESS_MAGNITUDE = {
    "angry":     [0.304, 1.734, -0.517, 1.329, -0.912, 1.226, -0.748, 0.824, -0.257, 1.132, -0.138, -0.808],
    "calm":      [-0.39, -1.827, -0.346, -1.043, -0.145, -1.052, -0.49, -0.203, -0.262, -0.844, -0.622, 0.489],
    "disgusted": [1.242, 1.214, 0.647, 1.518, 0.761, 1.395, 1.093, 0.87, 0.587, 0.57, 1.398, 0.76],
    "fearful":   [-1.441, -1.656, -1.628, -1.857, -1.072, -1.752, -2.262, -1.342, -1.423, -1.046, -2.641, -2.088],
    "happy":     [-0.018, -0.572, -0.081, -0.318, 0.348, -0.124, 0.34, -0.099, 0.057, -0.388, 0.194, 0.382],
    "neutral":   [-0.331, -0.933, -0.039, -0.823, 0.388, -0.761, -0.2, -0.514, -0.219, -0.839, -0.2, 0.141],
    "sad":       [-1.032, -0.303, -0.473, -0.455, -0.977, -0.351, -0.72, 0.441, -0.145, -0.142, -0.671, -0.012],
    "surprised": [1.885, 0.954, 1.228, 1.32, 1.447, 1.334, 1.492, 0.587, 0.601, 0.619, 1.808, 0.916],
}

# --- Table 4: clustering metrics, all concepts (RAVDESS) --------------------
# n_clusters -> (euclid ARI, cosine ARI, euclid silhouette, cosine silhouette)
TABLE_4_RAVDESS_CLUSTERING = {
    2: (0.159, 0.167, 0.447, 0.515), 3: (0.230, 0.284, 0.341, 0.516),
    4: (0.382, 0.414, 0.412, 0.442), 5: (0.388, 0.456, 0.381, 0.438),
    6: (0.377, 0.490, 0.325, 0.431), 7: (0.405, 0.457, 0.329, 0.393),
    8: (0.413, 0.491, 0.311, 0.391), 9: (0.402, 0.542, 0.293, 0.278),
    10: (0.415, 0.485, 0.294, 0.271),
}

# --- Table 7: true-label distribution over centroid-per-label clusters ------
# Diagonal only (fraction of each label assigned to its own centroid).
TABLE_7_RAVDESS_DIAGONAL = {
    "angry": 0.973, "calm": 0.836, "disgusted": 0.889, "fearful": 1.000,
    "happy": 0.188, "neutral": 0.381, "sad": 0.497,
    # 'surprised' row in the book sums to 0.164 across the printed columns; the
    # surprised column itself is cut off by the page margin.
}
