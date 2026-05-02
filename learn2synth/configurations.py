"""
configurations.py
=================
Central configuration for FCD SynthSeg training.
All dataset paths, file names, CSV path, and per-subject augmentation
type lists live here — import from this module everywhere else instead
of hardcoding values in the script.
"""
from learn2synth.custom_cc_synthseg import load_class_params_from_csv

# ── Dataset paths ─────────────────────────────────────────────────────────────
DEFAULT_FOLDER = '/kaggle/input/datasets/mazenmarwan122/final-dataset/Dataset001_FCDLesions/fcd'
OUTPUT_FOLDER = '/kaggle/working'
FLAIR_CLASS_PARAMS_CSV = '/kaggle/input/datasets/nancyabdelfattah/flair-stats-synthseg/flair_class_params_fcd_train.csv'
FLAIR_STATS_CSV = '/kaggle/input/datasets/nancyabdelfattah/flair-stats-synthseg/flair_stats_raw_fcd_train.csv'

# ── Per-subject file names ────────────────────────────────────────────────────
t1_file = 't1.nii'
flair_file = 'flair.nii'
roi_file = 'roi.nii'
label_file = 'labelmap.nii'
fusedmask_file = 'fusedmask.nii'  # 18-class SuperSynth label map (FCD lesion = 21)

# ── FLAIR class params — loaded once at import time ────────────────────────
FLAIR_CLASS_PARAMS = load_class_params_from_csv(FLAIR_CLASS_PARAMS_CSV)

# ── FCD subject lists by augmentation type ────────────────────────────────────
# A subject can appear in multiple lists; FCDDataset joins them with '+'
# (e.g. subject in both HYPER and TRANSMANTLE → aug_type = 'hyper+trans').

TRANSMANTLE_SUBJECTS = [
    5,   12,  17,  24,  33,  38,  41,  60,  65,  78,
    79,  83,  84,  85,  87,  101, 102, 106, 109, 116,
    119, 123, 129, 135, 137, 148, 149, 161, 163, 165,
    169
]

HYPER_SUBJECTS = [
    1,   3,   7,   13,  14,  15,  16,  18,  21,  22,
    23,  25,  26,  27,  29,  30,  35,  39,  40,  45,
    46,  49,  50,  51,  52,  53,  55,  56,  58,  63,
    69,  70,  73,  75,  77,  82,  89,  93,  97,  98,
    104, 110, 111, 112, 114, 118, 122, 124, 130, 133,
    138, 146, 152, 155, 158, 159, 160, 162, 166
]

BLUR_SUBJECTS = [
    26,  31,  37,  42,  44,  55,  65,  80,  81,  88,
    96,  105, 106, 115, 126, 127, 132, 143, 147, 150,
    152, 157, 163
]

THICKENING_SUBJECTS = [
    2,   7,   8,   10,  11,  13,  18,  19,  23,  24,
    28,  36,  43,  47,  49,  51,  54,  56,  57,  58,
    59,  63,  67,  68,  70,  71,  72,  76,  86,  89,
    90,  91,  94,  97,  102, 104, 113, 114, 116, 117,
    120, 122, 124, 131, 133, 138, 139, 140, 141, 146,
    149, 151, 153, 154, 155, 156, 158, 159, 160, 162,
    164, 167, 168, 170
]
INTENSITY_SUBJECTS = HYPER_SUBJECTS + TRANSMANTLE_SUBJECTS
