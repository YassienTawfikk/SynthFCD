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
DEFAULT_FOLDER = '/kaggle/input/datasets/yassienmohamed/bonnfcd-augmented-supersynth/fcd/'
OUTPUT_FOLDER = '/kaggle/working'
FLAIR_CLASS_PARAMS_CSV = '/kaggle/input/datasets/nancyabdelfattah/flair-stats-synthseg/flair_class_params_fcd_train.csv'
FLAIR_STATS_CSV = '/kaggle/input/datasets/nancyabdelfattah/flair-stats-synthseg/flair_stats_raw_fcd_train.csv'


# ── Per-subject file names ────────────────────────────────────────────────────
flair_file = 'flair.nii'
roi_file = 'roi.nii'
label_file = 'labelmap.nii'  # 18-class SuperSynth label map (FCD lesion = 21)
fusedmask_file = 'fusedmask.nii'  # kept for backwards compatibility


# ── FLAIR class params — loaded once at import time ────────────────────────
FLAIR_CLASS_PARAMS = load_class_params_from_csv(FLAIR_CLASS_PARAMS_CSV)


# ── FCD subject lists by augmentation type ────────────────────────────────────
# A subject can appear in multiple lists; FCDDataset joins them with '+'
# (e.g. subject in both HYPER and TRANSMANTLE → aug_type = 'hyper+trans').

TRANSMANTLE_SUBJECTS = [24, 33, 38, 60, 65, 78, 83, 87, 101, 109, 116, 123]

HYPER_SUBJECTS = [1, 3, 14, 15, 16, 18, 27, 40, 50, 53, 55, 58, 63, 73, 77, 89, 97, 98, 112, 122, 130, 133, 138, 146]

BLUR_SUBJECTS = [44, 55, 65, 80, 81, 105, 115, 126, 132]

THICKENING_SUBJECTS = [10, 18, 24, 43, 47, 58, 59, 63, 68, 71, 72, 76, 89, 90, 91, 97, 116, 120, 122, 131, 133, 138, 139, 140, 141, 146]

INTENSITY_SUBJECTS = HYPER_SUBJECTS + TRANSMANTLE_SUBJECTS
