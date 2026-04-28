"""
configurations.py
=================
Central configuration for FCD SynthSeg training.
All dataset paths, file names, CSV path, and per-subject augmentation
type lists live here — import from this module everywhere else instead
of hardcoding values in the script.
"""

# ── Dataset paths ─────────────────────────────────────────────────────────────
DEFAULT_FOLDER = '/kaggle/input/datasets/yassienmohamed/bonnfcd-augmented-supersynth/fcd/'
CSV_PATH = '/kaggle/input/datasets/nancyabdelfattah/flair-stats-synthseg/flair_stats_raw_fcd_train.csv'

# ── Per-subject file names ────────────────────────────────────────────────────
flair_file = 'flair.nii'
roi_file = 'roi.nii'
label_file = 'supersynth.nii'  # 18-class SuperSynth label map (FCD lesion = 21)
fusedmask_file = 'fusedmask.nii'  # kept for backwards compatibility

# ── FCD subject lists by augmentation type ────────────────────────────────────
# A subject can appear in multiple lists; FCDDataset joins them with '+'
# (e.g. subject in both HYPER and TRANSMANTLE → aug_type = 'hyper+trans').

TRANSMANTLE_SUBJECTS = [24, 33, 38, 60, 65, 78, 83, 87, 101, 109, 116, 123, 117, 154, 45, 118, 67, 137, 21, 111, 147, 93, 30, 163, 150, 88, 102, 153, 49, 143, 167,
                        5, 161, 127, 36, 158, 37, 39]

HYPER_SUBJECTS = [1, 3, 14, 15, 16, 18, 27, 40, 50, 53, 55, 58, 63, 73, 77, 89, 97, 98, 112, 122, 130, 133, 138, 146, 104, 113, 75, 152, 70, 19, 156, 7, 31, 84, 42,
                  94, 46, 13]

INTENSITY_SUBJECTS = HYPER_SUBJECTS + TRANSMANTLE_SUBJECTS

BLUR_SUBJECTS = [44, 55, 65, 80, 81, 105, 115, 126, 132, 17, 151, 164, 22, 124, 41, 157, 149, 23, 114, 2, 119, 165, 85, 166, 170, 51, 79, 106, 96, 110, 26, 168, 159,
                 155, 56, 52, 162, 169]

THICKENING_SUBJECTS = [10, 18, 24, 43, 47, 58, 59, 63, 68, 71, 72, 76, 89, 90, 91, 97, 116, 120, 122, 131, 133, 138, 139, 140, 141, 146, 12, 129, 25, 160, 28, 35,
                       54, 69, 82, 11, 29]
