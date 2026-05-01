"""FCD SynthSeg – 7-class FCD lesion segmentation (background + 5 tissue groups + FCD lesion)."""
# ── Standard library ────────────────────────────────────────────────────────
import os
import sys
import glob
import math
import random
import shutil
import datetime
import traceback
from typing import Sequence, Optional

from os import path, makedirs
from random import shuffle

# ── Third-party libraries ───────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import nibabel as nib

import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from torch.utils.data import Dataset, DataLoader
from torchmetrics.segmentation import DiceScore as dice_compute

import cornucopia as cc
from cornucopia import SynthFromLabelTransform, IntensityTransform
from cornucopia.special import IdentityTransform

# NOTE: GaussianSmooth is currently unused → remove if not needed
# from monai.transforms import GaussianSmooth


# ── Project path setup ──────────────────────────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# ── Local modules (learn2synth) ─────────────────────────────────────────────
from learn2synth.networks import UNet, SegNet
from learn2synth.train import SynthSeg
from learn2synth.losses import (
    DiceLoss, LogitMSELoss, CatLoss, CatMSELoss,
    DiceCELoss, FocalTverskyLoss,
)
from learn2synth import optim
from learn2synth.parameters import FCDParameterCalculator
from learn2synth.augmentations import FCDAugmentations

from learn2synth.custom_cc_synthseg import (
    SynthFromLabelTransform as CustomSynthFromLabelTransform,
)

# ── Configuration (single source of truth) ──────────────────────────────────
from learn2synth.configurations import (
    DEFAULT_FOLDER,
    OUTPUT_FOLDER,
    FLAIR_STATS_CSV,
    FLAIR_CLASS_PARAMS,
    flair_file,
    roi_file,
    label_file,
    INTENSITY_SUBJECTS,
    TRANSMANTLE_SUBJECTS,
    HYPER_SUBJECTS,
    BLUR_SUBJECTS,
    THICKENING_SUBJECTS,
)


# ── FCDDataset ────────────────────────────────────────────────────────────────
# Returns un-augmented volumes plus random augmentation configurations.
# Actual GPU synthesis happens inside Model.synthesize_batch

class FCDDataset(Dataset):
    def __init__(
            self,
            ndim,
            label_paths,
            flair_paths,
            roi_paths,
            fcd_intensity_range=(0.1, 0.5),
            fcd_tail_length_range=(20, 50),
            blur_sigma_range=(0.7, 1.7),
            zoom_f_range=(0.2, 0.4),
            hyper_sigma_range=(0.0, 0.3),
            trans_sigma_range=(0.0, 0.3)
    ):
        """
        Args:
            ndim: Dimensions of the input data.
            label_paths, flair_paths, roi_paths: Paths to the respective NIfTI volumes.
            fcd_intensity_range: Range for synthetic lesion intensity factors.
            fcd_tail_length_range: Range for the length of the transmantle tail.
            blur_sigma_range: Range for Gaussian blur augmentation.
            zoom_f_range: Range for cortical thickening (zoom) factor.
            hyper_sigma_range: Range for gray matter hyperintensity noise.
            trans_sigma_range: Range for transmantle signal intensity noise.
        """
        self.ndim = ndim
        self.fcd_intensity_range = fcd_intensity_range
        self.fcd_tail_length_range = fcd_tail_length_range

        # Store augmentation hyperparameters for external configurability
        self.blur_sigma_range = blur_sigma_range
        self.zoom_f_range = zoom_f_range
        self.hyper_sigma_range = hyper_sigma_range
        self.trans_sigma_range = trans_sigma_range

        self.items = []

        # Initialize stateless utility once to minimize instantiation overhead during loading
        self._calc = FCDParameterCalculator()

        for label_path, flair_path, roi_path in zip(label_paths, flair_paths, roi_paths):
            subject_num = self._calc.get_subj_num(os.path.dirname(label_path))
            aug_matches = []

            # Determine specific augmentation types based on subject-specific manifests
            if subject_num in BLUR_SUBJECTS:        aug_matches.append('blur')
            if subject_num in THICKENING_SUBJECTS:  aug_matches.append('zoom')
            if subject_num in HYPER_SUBJECTS:       aug_matches.append('hyper')
            if subject_num in TRANSMANTLE_SUBJECTS: aug_matches.append('trans')

            aug_type = '+'.join(aug_matches) if aug_matches else 'combo'
            self.items.append((label_path, flair_path, roi_path, aug_type))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        label_path, flair_path, roi_path, aug_type = self.items[idx]

        # --- Volume Loading (I/O) ---
        flair_arr = nib.load(flair_path).get_fdata()
        label_arr = nib.load(label_path).get_fdata().astype(int)
        roi_arr = nib.load(roi_path).get_fdata().astype(int)

        # --- Spatial Standardization ---
        # Resample ROI and labels to match FLAIR space using shared utility instance
        if flair_arr.shape != roi_arr.shape:
            roi_arr = (self._calc.resample_to_target(roi_arr, flair_arr.shape, True) > 0.5).astype(int)
        if flair_arr.shape != label_arr.shape:
            label_arr = self._calc.resample_to_target(label_arr, flair_arr.shape, True).astype(int)

        # --- Tensor Conversion ---
        # Data is prepared for the CNN pipeline (C, H, W, D format)
        label_tensor = torch.as_tensor(label_arr, dtype=torch.int64).unsqueeze(0)
        flair_tensor = torch.as_tensor(flair_arr, dtype=torch.float32).unsqueeze(0)
        roi_tensor = torch.as_tensor(roi_arr, dtype=torch.int64).unsqueeze(0)

        # --- Parameter Exposure ---
        # Construct augmentation parameters from instance ranges.
        # These are passed as tensors to ensure consistency across worker processes.
        aug_params = {
            'int_factor_min': torch.tensor(self.fcd_intensity_range[0], dtype=torch.float32),
            'int_factor_max': torch.tensor(self.fcd_intensity_range[1], dtype=torch.float32),
            'tail_length_min': torch.tensor(self.fcd_tail_length_range[0], dtype=torch.long),
            'tail_length_max': torch.tensor(self.fcd_tail_length_range[1], dtype=torch.long),

            'blur_sigma_min': torch.tensor(self.blur_sigma_range[0], dtype=torch.float32),
            'blur_sigma_max': torch.tensor(self.blur_sigma_range[1], dtype=torch.float32),

            'zoom_f_min': torch.tensor(self.zoom_f_range[0], dtype=torch.float32),
            'zoom_f_max': torch.tensor(self.zoom_f_range[1], dtype=torch.float32),

            'hyper_sigma_min': torch.tensor(self.hyper_sigma_range[0], dtype=torch.float32),
            'hyper_sigma_max': torch.tensor(self.hyper_sigma_range[1], dtype=torch.float32),

            'trans_sigma_min': torch.tensor(self.trans_sigma_range[0], dtype=torch.float32),
            'trans_sigma_max': torch.tensor(self.trans_sigma_range[1], dtype=torch.float32),
        }

        return {
            'label_t': label_tensor,
            'flair_t': flair_tensor,
            'roi_t': roi_tensor,
            'aug_type': aug_type,
            'subject_id': os.path.basename(os.path.dirname(label_path)),
            **aug_params,
        }

# ── FCDDataModule ─────────────────────────────────────────────────────────────
#
#  Source-aware split policy
#  ─────────────────────────
#  Training  : raw/ + generated/   (all available subjects)
#  Validation: raw/ only           (real subjects with ground-truth labels)
#
#  The split is enforced structurally — by scanning the two source directories
#  independently — rather than post-hoc via a fraction, so generated subjects
#  can never leak into validation regardless of ordering or preshuffle.
#
#  `eval` controls what fraction of *raw* subjects are held out for validation.
#  The remaining raw subjects join training alongside the entire generated pool.
#
#  Configurable via constructor flags:
#    train_subdir      (default "train")       — root under dataset_path
#    raw_subdir        (default "raw")         — val-eligible subfolder
#    extra_subdirs     (default ["generated"]) — train-only subfolders
#    use_extra_data    (default False)          — set True to train on raw + extra
#    val_from_raw_only (default False)          — set True to take validation from raw only
# ──────────────────────────────────────────────────────────────────────────────

class FCDDataModule(pl.LightningDataModule):
    def __init__(self,
                 ndim: int = 3,
                 dataset_path: str = DEFAULT_FOLDER,
                 eval: float = 0.04,
                 preshuffle: bool = False,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 train_subdir: str = 'train',
                 raw_subdir: Optional[str] = 'raw',
                 extra_subdirs: Optional[list] = None,
                 use_extra_data: bool = False):
        super().__init__()

        # --- Config ---
        self.ndim = ndim
        self.dataset_path = dataset_path
        self.eval_frac = eval
        self.preshuffle = preshuffle
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.use_extra_data = use_extra_data

        # --- Resolve directory layout ---
        # If raw_subdir is None, data lives directly in train_subdir — no subfolders,
        # no extra data possible, so use_extra_data is forced False.
        if raw_subdir is None:
            self.use_extra_data = False
            self.val_from_raw_only = False
            print("[FCDDataModule] raw_subdir=None — scanning train_subdir directly, use_extra_data forced False.")
        elif extra_subdirs is None:
            extra_subdirs = ['generated']

        train_root = path.join(dataset_path, train_subdir)
        raw_root = train_root if raw_subdir is None else path.join(train_root, raw_subdir)
        extra_roots = [path.join(train_root, s) for s in extra_subdirs] if extra_subdirs else []

        # --- Helper: scan one directory for valid (label, flair, roi) triplets ---
        def _scan(root: str) -> tuple:
            subject_folders = sorted(glob.glob(path.join(root, 'sub-*')))
            label_paths, flair_paths, roi_paths = [], [], []
            dropped = 0
            for subject_dir in subject_folders:
                label_path = path.join(subject_dir, label_file)
                flair_path = path.join(subject_dir, flair_file)
                roi_path = path.join(subject_dir, roi_file)
                if all(path.exists(x) for x in [label_path, flair_path, roi_path]):
                    label_paths.append(label_path)
                    flair_paths.append(flair_path)
                    roi_paths.append(roi_path)
                else:
                    dropped += 1
            if dropped:
                print(f"[FCDDataModule] WARNING: {dropped} incomplete triplets dropped in {root}")
            else:
                print(f"[FCDDataModule] {len(label_paths)} subjects loaded from {root}")
            return label_paths, flair_paths, roi_paths

        # --- Scan raw (val-eligible) subjects ---
        raw_label_paths, raw_flair_paths, raw_roi_paths = _scan(raw_root)
        assert len(raw_label_paths) > 0, (
            f"[FCDDataModule] Fatal: 0 valid triplets in '{raw_root}'. "
            "Check path and file names."
        )

        # --- Scan extra (train-only) subjects ---
        extra_label_paths, extra_flair_paths, extra_roi_paths = [], [], []
        if not self.use_extra_data:
            print("[FCDDataModule] use_extra_data=False — training on raw subjects only.")
        else:
            for extra_root in extra_roots:
                if path.isdir(extra_root):
                    e_labels, e_flairs, e_rois = _scan(extra_root)
                    extra_label_paths.extend(e_labels)
                    extra_flair_paths.extend(e_flairs)
                    extra_roi_paths.extend(e_rois)
                else:
                    print(f"[FCDDataModule] NOTE: extra subdir not found, skipping: {extra_root}")

        # --- Store split-ready pools ---
        # _raw_*   → split into val (first eval_frac) + train-raw (remainder)
        # _extra_* → always training only
        self._raw_label_paths = raw_label_paths
        self._raw_flair_paths = raw_flair_paths
        self._raw_roi_paths = raw_roi_paths

        self._extra_label_paths = extra_label_paths
        self._extra_flair_paths = extra_flair_paths
        self._extra_roi_paths = extra_roi_paths

        print(
            f"[FCDDataModule] Source summary: "
            f"{len(raw_label_paths)} raw, {len(extra_label_paths)} generated "
            f"→ val pool = raw only ({len(raw_label_paths)} subjects)"
        )

        # --- Compute FCD augmentation parameters from raw subjects only ---
        # Generated subjects have synthetic labels that may not reflect the
        # real intensity statistics expected by FCDParameterCalculator.
        print("[FCDDataModule] Computing FCD augmentation parameters…")
        self._calc = FCDParameterCalculator()
        self.fcd_intensity_range, self.fcd_tail_range = self._calc.calculate_fcd_parameters(            
            dataset_path=raw_root,
            label_file=label_file,
            flair_file=flair_file,
            roi_file=roi_file,
            intensity_subjects=INTENSITY_SUBJECTS,
            transmantle_subjects=TRANSMANTLE_SUBJECTS,
            auto_resample=True,
        )

    def setup(self, stage=None):
        if hasattr(self, '_setup_done'):
            return
        self._setup_done = True

        # Copy raw pool (and shuffle if requested) before splitting
        raw_label_paths = list(self._raw_label_paths)
        raw_flair_paths = list(self._raw_flair_paths)
        raw_roi_paths = list(self._raw_roi_paths)

        if self.preshuffle:
            combined = list(zip(raw_label_paths, raw_flair_paths, raw_roi_paths))
            shuffle(combined)
            raw_label_paths, raw_flair_paths, raw_roi_paths = map(list, zip(*combined))

        def _count(param, total):
            if isinstance(param, float): return int(math.ceil(total * param))
            if isinstance(param, int):   return param
            return 0

        # Split raw pool → val head + train tail
        n_val = _count(self.eval_frac, len(raw_label_paths))

        val_label_paths = raw_label_paths[:n_val]
        val_flair_paths = raw_flair_paths[:n_val]
        val_roi_paths = raw_roi_paths[:n_val]

        train_raw_label_paths = raw_label_paths[n_val:]
        train_raw_flair_paths = raw_flair_paths[n_val:]
        train_raw_roi_paths = raw_roi_paths[n_val:]

        # Training set = remaining raw + all extra (empty list if use_extra_data=False)
        train_label_paths = train_raw_label_paths + list(self._extra_label_paths)
        train_flair_paths = train_raw_flair_paths + list(self._extra_flair_paths)
        train_roi_paths = train_raw_roi_paths + list(self._extra_roi_paths)

        print(
            f"[FCDDataModule] Split: "
            f"train={len(train_label_paths)} ({len(train_raw_label_paths)} raw + {len(self._extra_label_paths)} generated), "
            f"val={len(val_label_paths)} (raw only)"
        )

        kw = dict(
            fcd_intensity_range=self.fcd_intensity_range,
            fcd_tail_length_range=self.fcd_tail_range,
        )

        self.train_ds = FCDDataset(self.ndim, train_label_paths, train_flair_paths, train_roi_paths, **kw)
        self.eval_ds = FCDDataset(self.ndim, val_label_paths, val_flair_paths, val_roi_paths, **kw)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.eval_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=self.num_workers > 0)


# ══════════════════════════════════════════════════════════════════════════════
#  SharedSynth  —  geometry + GMM forward pass, intensity kept separate
# ══════════════════════════════════════════════════════════════════════════════

class SharedSynth(torch.nn.Module):
    """
    GMM synthesis + label remapping for the FCD segmentation pipeline.

    Synthetic branch: label map → one-hot → GMM → synthetic FLAIR image
    Real branch:      FLAIR + label + ROI passed through unchanged (no_augs=True)

    Geometric deformation is disabled (no_augs=True in CustomSynthFromLabelTransform).
    IntensityTransform is intentionally excluded — applied downstream after FCD augmentations.
    """

    N_CLASSES = 18  # valid labels are 0..18 inclusive (19 values)

    def __init__(self, synth, target_labels=None):
        super().__init__()
        self.synth = synth
        self.target_labels = target_labels or []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_class_params(self, params: dict):
        """Swap the GMM's per-class intensity params (e.g. per-subject stats)."""
        gmm = getattr(self.synth, 'gmm', None)
        if gmm is not None and hasattr(gmm, 'class_params'):
            gmm.class_params = params
        else:
            print('[SharedSynth] Warning: set_class_params called but no GMM found.')

    def forward(self, slab, img, lab, roi):
        """
        Parameters
        ----------
        slab : (1, D, H, W) int64   — label map 0..18, used for GMM synthesis
        img  : (1, D, H, W) float32 — real FLAIR, passed through to real branch
        lab  : (1, D, H, W) int64   — real label map, passed through to real branch
        roi  : (1, D, H, W) int64   — binary ROI mask, passed through to real branch

        Returns
        -------
        simg     : (1, D, H, W) float32  — synthetic FLAIR from GMM
        slab_out : (1, D, H, W) int64    — remapped label map {0..5}
        rimg     : (1, D, H, W) float32  — real FLAIR (unchanged)
        rlab     : (1, D, H, W) int64    — real label remapped {0..5}
        rroi     : (1, D, H, W) int64    — ROI mask (unchanged)
        """
        img = img.float()

        # Route based on which synthesiser is attached.
        # cornucopia's SynthFromLabelTransform has make_final (legacy/non-FLAIR path).
        # CustomSynthFromLabelTransform (FLAIR path) is a plain nn.Module — no make_final.
        if hasattr(self.synth, 'make_final'):
            return self._forward_standard(slab, img, lab, roi)
        return self._forward_custom(slab, img, lab, roi)

    # ------------------------------------------------------------------
    # Forward paths
    # ------------------------------------------------------------------

    def _forward_standard(self, slab, img, lab, roi):
        """
        Legacy path — cornucopia SynthFromLabelTransform.
        Applies geometric deformation to all inputs using a shared random field.
        Used when modality != 'flair'.
        """
        final = self.synth.make_final(slab, 1)
        final.deform = final.deform.make_final(slab)

        simg, slab = final(slab)
        rimg, rlab, rroi = final.deform([img, lab, roi])
        rlab = final.postproc(rlab)

        return simg, slab, rimg, rlab, rroi

    def _forward_custom(self, slab, img, lab, roi):
        """
        FLAIR path — CustomSynthFromLabelTransform (no_augs=True).

        No geometric deformation is applied (disabled via no_augs=True).
        GMM samples per-class FLAIR intensities from the one-hot label map.
        IntensityTransform inside the synth is also disabled (donothing) —
        it is applied downstream after FCD augmentations.

        slab is remapped directly (not via one-hot round-trip) since
        no_augs=True means slab_oh.argmax() == slab anyway.
        """
        oh_slab = self._to_one_hot(slab)

        # synth applies GMM only (no deform, no intensity)
        # coreg=[img, lab, roi] passes real-branch inputs through unchanged
        simg, _, (rimg, rlab, rroi) = self.synth(oh_slab, coreg=[img, lab, roi])

        # Remap both label maps from sparse 0..18 → consecutive 0..5
        slab_out = self.remap_labels(slab)
        rlab_out = self.remap_labels(rlab)

        return simg, slab_out, rimg, rlab_out, rroi

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_one_hot(self, label_map: torch.Tensor) -> torch.Tensor:
        """
        Convert integer label map → one-hot float tensor (C, D, H, W).
        num_classes = N_CLASSES + 1 = 19 to accommodate labels 0..18 inclusive.
        """
        return (
            torch.nn.functional.one_hot(
                label_map.long().squeeze(0),
                num_classes=self.N_CLASSES + 1  # 19: covers labels 0..18
            )
            .permute(3, 0, 1, 2)
            .float()
        )

    def remap_labels(self, label_map: torch.Tensor) -> torch.Tensor:
        """
        Map sparse label values → consecutive model class indices.

        target_labels = [(1,), (2,), (3,), (4,), (18,)]
        Mapping:
            label 1  → class 1 (White Matter)
            label 2  → class 2 (Cerebral Cortex)
            label 3  → class 3 (Deep Gray Matter)
            label 4  → class 4 (CSF)
            label 18 → class 5 (WM-GM Separator)
            all else → class 0 (Background)
        """
        max_value = int(label_map.max().item()) + 1
        lut = torch.zeros(max_value, dtype=torch.long, device=label_map.device)

        for class_index, group in enumerate(self.target_labels, start=1):
            for value in group:
                if value < max_value:
                    lut[value] = class_index

        nb_classes = len(self.target_labels) + 1  # 6: background + 5 tissues
        return torch.clamp(lut[label_map.long()], 0, nb_classes - 1)

# ══════════════════════════════════════════════════════════════════════════════
#  Model  —  6-class grouped segmentation (brain structures + FCD lesion)
# ══════════════════════════════════════════════════════════════════════════════

class Model(pl.LightningModule):
    # ── Class-level label definitions — single source of truth ───────────────
    TARGET_LABELS = [
        (1,),   # White Matter
        (2,),   # Cerebral Cortex
        (3,),   # Deep Gray Matter
        (4,),   # CSF
        (18,),  # WM-GM Separator
    ]

    def __init__(
            self,
            ndim: int = 3,
            nb_classes: int = 7,                              # background + 5 tissues + FCD lesion
            seg_nb_levels: int = 6,
            seg_features: Sequence[int] = (16, 32, 64, 128, 256, 512),
            seg_activation: str = 'ReLU',
            seg_nb_conv: int = 2,
            seg_norm: Optional[str] = 'instance',
            loss: str = 'dice_ce',
            alpha: float = 1.0,
            optimizer: str = 'Adam',
            optimizer_options: Optional[dict] = None,
            time_limit_minutes: float = None,
            modality: str = '',
            flair_stats_csv: Optional[str] = FLAIR_STATS_CSV,
            n_best_batches: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.optimizer_name    = optimizer
        self.optimizer_options = dict(optimizer_options or {'lr': 1e-4})
        self.time_limit_minutes = time_limit_minutes
        self.alpha             = alpha
        self.flair_stats_csv   = flair_stats_csv
        self.target_labels     = self.TARGET_LABELS

        # ── Sub-modules ───────────────────────────────────────────────────────
        self.subject_params_cache = self._load_subject_params()
        seg_net       = self._build_seg_network(ndim, nb_classes, seg_features,seg_activation, seg_nb_levels, seg_nb_conv, seg_norm)
        synth         = self._build_synth(modality)
        loss_fn       = self._build_loss(loss)
        self.network  = SynthSeg(seg_net, synth, loss_fn)
        self.intensity_aug = self._build_intensity_aug(modality)
        self.fcd_aug  = FCDAugmentations()  # stateless utility — instantiated once

        # ── Metrics ───────────────────────────────────────────────────────────
        _m = dict(include_background=False, num_classes=nb_classes, input_format='index')
        self.val_dice     = dice_compute(average='micro', **_m)
        self.val_dice_fcd = dice_compute(average='none',  **_m)

        # ── Manual optimisation ───────────────────────────────────────────────
        self.automatic_optimization = False
        self.network.set_backward(self.manual_backward)

        # ── State ─────────────────────────────────────────────────────────────
        self.n_best_batches    = n_best_batches
        self._val_batch_cache  = []
        self._saved_aug_types  = set()

    # ══════════════════════════════════════════════════════════════════════════
    #  Lifecycle hooks
    # ══════════════════════════════════════════════════════════════════════════

    def on_train_start(self):
        # Wire the optimizer getter into SynthSeg so train_step() can call it.
        self.network.optimizers = self.optimizers

        # ── Model Architecture ────────────────────────────────────────────────
        seg = self.network.segnet
        print("\n" + "═" * 60)
        print("DEBUG: Model Architecture")
        print("═" * 60)
        print(f"  Backbone        : UNet")
        print(f"  seg_features    : {self.hparams.seg_features}")
        print(f"  seg_nb_levels   : {self.hparams.seg_nb_levels}")
        print(f"  seg_nb_conv     : {self.hparams.seg_nb_conv}")
        print(f"  seg_norm        : {self.hparams.seg_norm}")
        print(f"  nb_classes      : {self.hparams.nb_classes}")
        print(f"  modality        : '{self.hparams.modality}'")
        total_params = sum(p.numel() for p in seg.parameters())
        trainable = sum(p.numel() for p in seg.parameters() if p.requires_grad)
        print(f"  Total params    : {total_params:,}")
        print(f"  Trainable params: {trainable:,}")

        # ── Synthesis Pipeline ────────────────────────────────────────────────
        print("\nDEBUG: Synthesis Pipeline")
        print("─" * 60)
        print(f"  SharedSynth.synth type : {type(self.network.synth.synth).__name__}")
        gmm = getattr(self.network.synth.synth, 'gmm', None)
        print(f"  GMM type               : {type(gmm).__name__ if gmm else 'None'}")
        print(f"  GMM class_params keys  : {sorted(gmm.class_params.keys()) if gmm and hasattr(gmm, 'class_params') else 'N/A'}")
        print(f"  IntensityAug type      : {type(self.intensity_aug).__name__}")
        print(f"  FCDAugmentations       : {type(self.fcd_aug).__name__}")
        print(f"  Subject params cached  : {len(self.subject_params_cache)} subjects")


    # ══════════════════════════════════════════════════════════════════════════
    #  Private builders  (called only from __init__)
    # ══════════════════════════════════════════════════════════════════════════
    def _load_subject_params(self) -> dict:
        """Pre-load per-subject GMM parameters from CSV into a lookup cache."""
        cache = {}
        if not (self.flair_stats_csv and os.path.exists(self.flair_stats_csv)):
            return cache
        try:
            df = pd.read_csv(self.flair_stats_csv)
            df['subject'] = df['subject'].astype(str).str.strip()
            # range(19): covers classes 0–18 inclusive (WM-GM Separator = 18)
            default_keys = set(range(19)) | set(FLAIR_CLASS_PARAMS.keys())

            for subj in df['subject'].unique():
                params = {
                    int(r['class_id']): {
                        'mu':    (float(r['mu_lo']),    float(r['mu_hi'])),
                        'sigma': (float(r['sigma_lo']), float(r['sigma_hi'])),
                    }
                    for _, r in df[df['subject'] == subj].iterrows()
                }
                for cls in default_keys:
                    params.setdefault(
                        cls, FLAIR_CLASS_PARAMS.get(cls, {'mu': (0, 255), 'sigma': (0, 16)})
                    )
                cache[subj] = params

            print(f'[Model] Preloaded per-subject params for {len(cache)} subjects.')
        except Exception as exc:
            print(f'[Model] Warning: failed to parse CSV — {exc}')
        return cache

    def _build_seg_network(self, ndim, nb_classes, features, activation,
                           nb_levels, nb_conv, norm):
        backbone = UNet(ndim, nb_features=features, activation=activation,
                        nb_levels=nb_levels, nb_conv=nb_conv, norm=norm)
        return SegNet(ndim, 1, nb_classes, backbone=backbone, activation=None)

    def _build_synth(self, modality: str) -> SharedSynth:
        if modality == 'flair':
            raw = CustomSynthFromLabelTransform(
                num_ch=1, class_params=FLAIR_CLASS_PARAMS, use_per_class_gmm=True,
                gmm_fwhm=5, bias=5, gamma=0.4, motion_fwhm=2, resolution=4,
                snr=15, gfactor=3, rotation=10, shears=0.008, zooms=0.10,
                elastic=0.03, elastic_nodes=8, order=3, no_augs=True,
            )
        else:
            raw = SynthFromLabelTransform(
                one_hot=False, target_labels=self.target_labels,
                elastic=0.05, elastic_nodes=10, rotation=15, shears=0.012,
                zooms=0.15, resolution=5, motion_fwhm=2.0, snr=10,
                gmm_fwhm=10, gamma=0.5, bias=7, bias_strength=0.5,
            )
            raw.intensity = IdentityTransform()

        return SharedSynth(raw, target_labels=self.target_labels)

    def _build_loss(self, loss: str):
        options = {
            'dice':          lambda: DiceLoss(activation='Softmax'),
            'logitmse':      lambda: LogitMSELoss(),
            'cat':           lambda: CatLoss(activation='Softmax'),
            'catmse':        lambda: CatMSELoss(activation='Softmax'),
            'dice_ce':       lambda: DiceCELoss(activation='Softmax'),
            'focal_tversky': lambda: FocalTverskyLoss(activation='Softmax'),
        }
        if loss not in options:
            raise ValueError(f"Unsupported loss '{loss}'. Choose from: {list(options)}")
        return options[loss]()

    def _build_intensity_aug(self, modality: str):
        if modality == 'flair':
            return IntensityTransform(
                bias=5, gamma=0.4, motion_fwhm=2, resolution=4,
                snr=15, gfactor=3, order=3,
            )
        return IntensityTransform(
            bias=7, bias_strength=0.2, gamma=0.3, motion_fwhm=3,
            resolution=4, snr=20, gfactor=2, order=3,
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  GMM subject param switching
    # ══════════════════════════════════════════════════════════════════════════

    def _set_subject_params(self, subject_id: Optional[str]):
        """Swap GMM class_params for the current subject (falls back to global)."""
        params = (
            self.subject_params_cache[subject_id]
            if subject_id and subject_id in self.subject_params_cache
            else FLAIR_CLASS_PARAMS
        )
        self.network.synth.set_class_params(params)
    # ══════════════════════════════════════════════════════════════════════════
    #  Augmentation pipeline
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _parse_aug_choices(aug_type: str) -> list:
        if aug_type == 'combo':
            return random.sample(['blur', 'zoom', 'hyper', 'trans'], random.randint(1, 4))
        return aug_type.split('+')

    def _apply_fcd_augmentations(self, img: torch.Tensor, roi: torch.Tensor,
                                 choices: list, params: dict) -> torch.Tensor:
        """Apply the FCD augmentation chain. Returns a new tensor."""
        for ch in choices:
            if ch == 'zoom':
                img = self.fcd_aug.apply_roi_thickening(
                    img, roi, zoom_range=params['zoom_f'])
            elif ch in ('hyper', 'trans'):
                img = self.fcd_aug.apply_roi_augmentations_hyperintensity(
                    img, roi,
                    intensity_range=params['int_rng'],
                    sigma_range=params['hyper_sigma'] if ch == 'hyper' else params['trans_sigma'],
                )
            elif ch == 'blur':
                img = self.fcd_aug.apply_roi_augmentations_blured(
                    img, roi, sigma_range=params['blur_sigma'])
        return img

    def _maybe_save_aug_sample(self, aug_img: torch.Tensor, aug_mask: torch.Tensor,
                               aug_type: str):
        """Save one NIfTI sample per aug_type for debugging (fires once per type)."""
        if aug_type in self._saved_aug_types:
            return
        self._saved_aug_types.add(aug_type)
        save_dir = os.path.join(OUTPUT_FOLDER, 'saved_augs')
        os.makedirs(save_dir, exist_ok=True)
        nib.save(
            nib.Nifti1Image(aug_img.detach().cpu().numpy(), np.eye(4)),
            os.path.join(save_dir, f'subj_aug_{aug_type}_img.nii.gz'),
        )
        nib.save(
            nib.Nifti1Image(aug_mask.detach().cpu().numpy().astype(np.uint8), np.eye(4)),
            os.path.join(save_dir, f'subj_aug_{aug_type}_mask.nii.gz'),
        )

    def _process_single_sample(self, batch: dict, i: int):
        """
        Full single-sample synthesis pipeline:
          SharedSynth → FCD aug → IntensityTransform → label fusion.
        Returns (aug_image, aug_mask, real_image, real_mask) or None if skipped.
        """
        label_t    = batch['label_t'][i]
        flair_t    = batch['flair_t'][i].float()
        roi_t      = batch['roi_t'][i]
        aug_type   = batch['aug_type'][i]
        subject_id = batch.get('subject_id', [None] * len(batch['label_t']))[i]

        aug_params = {
            'int_rng':     (batch['int_factor_min'][i].item(),  batch['int_factor_max'][i].item()),
            'blur_sigma':  (batch['blur_sigma_min'][i].item(),  batch['blur_sigma_max'][i].item()),
            'zoom_f':      (batch['zoom_f_min'][i].item(),      batch['zoom_f_max'][i].item()),
            'hyper_sigma': (batch['hyper_sigma_min'][i].item(), batch['hyper_sigma_max'][i].item()),
            'trans_sigma': (batch['trans_sigma_min'][i].item(), batch['trans_sigma_max'][i].item()),
        }

        if label_t.sum() == 0 or torch.isnan(label_t.float()).any():
            return None

        self._set_subject_params(subject_id)

        # Step 1: SharedSynth — geometry + GMM, no intensity transform
        simg, slab, rimg, rlab, rroi = self.network.synth(label_t, flair_t, label_t, roi_t)
        simg_3d = simg.squeeze(0).float()
        slab_3d = slab.squeeze(0).long()
        rroi_3d = (rroi.squeeze(0) > 0).long()

        # Step 2: FCD appearance augmentations (synthetic branch only)
        choices = self._parse_aug_choices(aug_type)
        aug_img = self._apply_fcd_augmentations(simg_3d.clone(), rroi_3d, choices, aug_params)

        # Step 3: Debug NIfTI save — fires once per aug_type
        self._maybe_save_aug_sample(aug_img, rroi_3d, aug_type)

        # Step 4: Standalone IntensityTransform (bias field, gamma, noise, resolution)
        aug_out        = self.intensity_aug(aug_img.float().unsqueeze(0))
        aug_image_item = aug_out[0] if isinstance(aug_out, (list, tuple)) else aug_out

        # Step 5: Fuse FCD lesion (ROI → class 6) into both label maps
        slab_with_fcd = slab_3d.clone()
        slab_with_fcd[rroi_3d > 0] = 6

        rlab_with_fcd = rlab.long().squeeze(0).clone()
        rlab_with_fcd[rroi_3d > 0] = 6

        return (
            aug_image_item,              # (1, D, H, W)  synthetic FLAIR — normalized [0,1]
            slab_with_fcd.unsqueeze(0),  # (1, D, H, W)  synthetic target {0–6}
            rimg.float(),                # (1, D, H, W)  real FLAIR — raw scale [~0, 245]
            rlab_with_fcd.unsqueeze(0),  # (1, D, H, W)  real target {0–6}
        )

    def synthesize_batch(self, batch: dict):
        """Run the synthesis pipeline for every sample. Returns stacked tensors."""
        results     = []
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            for i in range(len(batch['label_t'])):
                out = self._process_single_sample(batch, i)
                if out is not None:
                    results.append(out)
            if not results:
                return None

        aug_images, aug_masks, real_images, real_masks = zip(*results)
        return (
            torch.stack(aug_images).float(),
            torch.stack(aug_masks).long(),
            torch.stack(real_images).float(),
            torch.stack(real_masks).long(),
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  Training
    # ══════════════════════════════════════════════════════════════════════════

    def training_step(self, batch, batch_idx):
        # Periodically free CUDA cache to prevent fragmentation over long runs
        if self.trainer.current_epoch % 10 == 0 and batch_idx == 0:
            torch.cuda.empty_cache()

        result = self.synthesize_batch(batch)
        if result is None:
            return None
        aug_image, aug_mask, real_image, real_mask = result

        loss_synth, loss_real = self.network.train_step(
            aug_image, aug_mask, real_image, real_mask)

        loss = loss_synth + self.alpha * loss_real
        self.log('train_loss', loss, prog_bar=True)
        return loss

    # ══════════════════════════════════════════════════════════════════════════
    #  Validation
    # ══════════════════════════════════════════════════════════════════════════

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            result = self.synthesize_batch(batch)
            if result is None:
                return None
            aug_image, aug_mask, real_image, real_mask = result

            loss_synth, loss_real, pred_synth, pred_real = self.network.eval_for_plot(
                aug_image, aug_mask, real_image, real_mask)

        pred_labels   = pred_real.cpu().argmax(dim=1)
        target_labels = real_mask.cpu().squeeze(1).long()

        self.val_dice.update(pred_labels, target_labels)
        self.val_dice_fcd.update(pred_labels, target_labels)

        loss = loss_synth + self.alpha * loss_real
        self.log('eval_loss', loss, prog_bar=True)

        # Cache for top-N best-batch NIfTI diagnostics
        self._val_batch_cache.append({
            'pred_synth':    pred_synth.cpu(),
            'pred_labels':   pred_labels,
            'aug_image':     aug_image.cpu(),
            'real_image':    real_image.cpu(),
            'aug_mask':      aug_mask.cpu(),
            'target_labels': target_labels,
            'score':         -loss.item(),   # lower loss = higher score
            'batch_idx':     batch_idx,
        })
        self._val_batch_cache.sort(key=lambda x: x['score'], reverse=True)
        self._val_batch_cache = self._val_batch_cache[:self.n_best_batches]

        return loss

    def _log_val_diagnostics(self, pred_synth, pred_labels, aug_image,
                             real_image, aug_mask, real_labels, suffix=''):
        """Log class-count scalars and save NIfTI samples every 20 epochs."""
        pred_synth_argmax = pred_synth[0].argmax(dim=0)
        pred_real_argmax  = pred_labels[0]
        self.log('pred_synth_num_classes',
                 float(len(torch.unique(pred_synth_argmax))), prog_bar=False)
        self.log('pred_real_num_classes',
                 float(len(torch.unique(pred_real_argmax))),  prog_bar=False)

        if self.trainer.current_epoch % 20 != 0:
            return

        base_dir = self.trainer.log_dir or self.trainer.default_root_dir
        img_root = os.path.join(base_dir, 'images')
        makedirs(img_root, exist_ok=True)
        print(f'\n[Saving] NIfTI diagnostics — Epoch {self.trainer.current_epoch}'
              f' {suffix} → {img_root}')

        p = f'{img_root}/epoch-{self.trainer.current_epoch:04d}'
        save(pred_synth_argmax,                  f'{p}_{suffix}_synth-pred.nii.gz')
        save(pred_real_argmax,                   f'{p}_{suffix}_real-pred.nii.gz')
        save(aug_image[0].squeeze(0),            f'{p}_{suffix}_synth-image.nii.gz')
        save(real_image[0].squeeze(0),           f'{p}_{suffix}_real-image.nii.gz')
        save(aug_mask[0].squeeze(0).to(torch.uint8),  f'{p}_{suffix}_synth-ref.nii.gz')
        save(real_labels[0].to(torch.uint8),     f'{p}_{suffix}_real-ref.nii.gz')

    def on_validation_epoch_end(self):
        # Save NIfTI diagnostics for the best N cached batches
        for i, bd in enumerate(self._val_batch_cache):
            self._log_val_diagnostics(
                bd['pred_synth'].to(self.device),
                bd['pred_labels'],
                bd['aug_image'].to(self.device),
                bd['real_image'].to(self.device),
                bd['aug_mask'].to(self.device),
                bd['target_labels'],
                suffix=f'_best{i}_batch{bd["batch_idx"]}',
            )
        self._val_batch_cache = []

        dice_epoch = self.val_dice.compute()
        dice_per_cls = self.val_dice_fcd.compute()
        dice_fcd = dice_per_cls[5] if len(dice_per_cls) > 5 else torch.tensor(0.0)

        self.log('val_dice',     dice_epoch, prog_bar=True)
        self.log('val_dice_fcd', dice_fcd,   prog_bar=False)

        tl = self.trainer.callback_metrics.get('train_loss', -1)
        el = self.trainer.callback_metrics.get('eval_loss',  -1)
        print(f"\n{'=' * 40}")
        print(f"EPOCH {self.trainer.current_epoch} SUMMARY:")
        print(f"  Train Loss    : {float(tl):.4f}")
        print(f"  Eval Loss     : {float(el):.4f}")
        print(f"  DICE SCORE    : {dice_epoch:.4f}")
        print(f"  DICE FCD (c6) : {dice_fcd:.4f}")
        print(f"{'=' * 40}\n")

        # Log current LR from the scheduler Lightning manages
        current_lr = self.optimizers().param_groups[0]['lr']
        print(f'  LR            : {current_lr:.2e}')

        self.val_dice.reset()
        self.val_dice_fcd.reset()


    # ══════════════════════════════════════════════════════════════════════════
    #  Optimiser / callbacks / inference
    # ══════════════════════════════════════════════════════════════════════════

    def configure_optimizers(self):
        opt_cls = getattr(optim, self.optimizer_name)
        optimizer = opt_cls(self.network.segnet.parameters(),
                            **(self.optimizer_options or {}))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "eval_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def configure_callbacks(self):
        return [TimeLimitCallback(self.time_limit_minutes)] if self.time_limit_minutes else []

    def forward(self, x):
        return self.network.segnet(x)


# ── Helper Functions ──────────────────────────────────────────────────────────

def save(dat, fname):
    dat = dat.detach().cpu().numpy()
    h = nib.Nifti1Header()
    h.set_data_dtype(dat.dtype)
    nib.save(nib.Nifti1Image(dat, np.eye(4), h), fname)


# ══════════════════════════════════════════════════════════════════════════════
#  TimeLimitCallback
# ──────────────────────────────────────────────────────────────────────────────
#  Stops training cleanly after N minutes (epoch boundary, not mid-batch).
#  Reads L2S_TIME_LIMIT_MINUTES from the environment at on_train_start so
#  resuming with a new time budget always takes effect without touching the
#  checkpoint's baked-in hparams.
# ══════════════════════════════════════════════════════════════════════════════
class TimeLimitCallback(Callback):
    def __init__(self, limit_minutes):
        self.limit_minutes_default = limit_minutes
        self.limit_minutes = None
        self.start_time = None

    def on_train_start(self, trainer, pl_module):
        env_val = os.environ.get('L2S_TIME_LIMIT_MINUTES')
        if env_val is not None:
            self.limit_minutes = float(env_val)
            print(f"\n[TimeLimit] Limit set from environment: {self.limit_minutes} mins.")
        else:
            self.limit_minutes = self.limit_minutes_default
            print(f"\n[TimeLimit] Limit set from model hparams: {self.limit_minutes} mins.")
        self.start_time = datetime.datetime.now()
        print(f"[TimeLimit] Training started at {self.start_time}.")

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.limit_minutes and self.start_time:
            elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
            if elapsed > self.limit_minutes * 60:
                print(f"\n[TimeLimit] Time limit reached "
                      f"({elapsed / 60:.1f} > {self.limit_minutes} mins). "
                      f"Stopping after this epoch.")
                trainer.should_stop = True


# ══════════════════════════════════════════════════════════════════════════════
#  LossGraphCallback
# ──────────────────────────────────────────────────────────────────────────────
#  Dual-axis plot: Loss (left) + Dice (right), saved as training_plot.png
#  after every validation epoch. Silent on errors — never blocks training.
# ══════════════════════════════════════════════════════════════════════════════
class LossGraphCallback(Callback):
    """Dual-axis loss + dice plot, one PNG per validation epoch."""

    def on_validation_epoch_end(self, trainer, pl_module):
        try:
            if not trainer.log_dir:
                return
            metrics_path = os.path.join(trainer.log_dir, "metrics.csv")
            plot_path = os.path.join(trainer.log_dir, "training_plot.png")
            if not os.path.exists(metrics_path):
                return
            try:
                metrics = pd.read_csv(metrics_path)
            except Exception as e:
                print(f"[LossGraph] Failed to read metrics.csv at epoch "
                      f"{trainer.current_epoch}: {type(e).__name__}: {e}")
                return

            epoch_metrics = metrics.groupby("epoch").mean()

            fig, ax1 = plt.subplots(figsize=(10, 6))

            if 'train_loss' in epoch_metrics:
                ax1.plot(epoch_metrics.index, epoch_metrics['train_loss'],
                         label='Train Loss', color='blue', linestyle='-', alpha=0.7)
            if 'eval_loss' in epoch_metrics:
                ax1.plot(epoch_metrics.index, epoch_metrics['eval_loss'],
                         label='Val Loss', color='red', linestyle='--')

            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_ylim(bottom=0)
            ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

            ax2 = ax1.twinx()
            if 'val_dice' in epoch_metrics:
                ax2.plot(epoch_metrics.index, epoch_metrics['val_dice'],
                         label='Val Dice', color='green', linewidth=2)
            if 'val_dice_fcd' in epoch_metrics:
                ax2.plot(epoch_metrics.index, epoch_metrics['val_dice_fcd'],
                         label='Val Dice FCD (c6)', color='orange',
                         linewidth=2, linestyle='--')
            ax2.set_ylabel('Dice')
            ax2.set_ylim(0, 1)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(
                lines1 + lines2, labels1 + labels2,
                loc='upper center', bbox_to_anchor=(0.5, -0.12),
                ncol=4, frameon=True,
            )

            fig.suptitle(f'Training Metrics (Epoch {trainer.current_epoch})')
            fig.tight_layout(rect=[0, 0.08, 1, 0.95])

            try:
                plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            finally:
                plt.close()
        except Exception as e:
            print(f"[LossGraph] ❌ Error at epoch {trainer.current_epoch}: "
                  f"{type(e).__name__}: {e}")
            traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
#  EveryEpochCheckpointCallback
# ──────────────────────────────────────────────────────────────────────────────
#  Writes checkpoints/<filename> unconditionally after every training epoch.
#  Does not depend on ModelCheckpoint, metric improvements, or save_last logic.
#  This is the canonical file used for resuming.
#
#  Uses trainer.save_checkpoint() directly — bypasses all of ModelCheckpoint's
#  link/top-k/metric-gating logic.
# ══════════════════════════════════════════════════════════════════════════════
class EveryEpochCheckpointCallback(Callback):
    def __init__(self, filename="resume.ckpt"):
        self.filename = filename

    def on_train_epoch_end(self, trainer, pl_module):
        ckpt_dir = os.path.join(trainer.default_root_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, self.filename)
        try:
            trainer.save_checkpoint(ckpt_path)
            size_mb = os.path.getsize(ckpt_path) / 1e6
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(ckpt_path))
            print(f"[EveryEpoch] ✅ epoch={trainer.current_epoch} → "
                  f"{ckpt_path} ({size_mb:.1f} MB, mtime {mtime})")
        except Exception as e:
            print(f"[EveryEpoch] ❌ Save FAILED at epoch "
                  f"{trainer.current_epoch}: {type(e).__name__}: {e}")
            traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
#  CheckpointTraceCallback
# ──────────────────────────────────────────────────────────────────────────────
#  Diagnostic callback — verifies resume.ckpt and last.ckpt behave as expected
#  each epoch. Fires after EveryEpochCheckpointCallback so resume.ckpt is
#  already written when we inspect it.
# ══════════════════════════════════════════════════════════════════════════════
class CheckpointTraceCallback(Callback):

    def on_validation_epoch_start(self, trainer, pl_module):
        print(f"\n[CKPT TRACE] === Epoch {trainer.current_epoch}: validation starting ===")

    def on_validation_epoch_end(self, trainer, pl_module):
        _, _, free = shutil.disk_usage(OUTPUT_FOLDER)
        print(f"[CKPT TRACE] Epoch {trainer.current_epoch}: validation hooks running. "
              f"Disk free={free / 1e9:.2f}GB, should_stop={trainer.should_stop}")

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        mc = next((cb for cb in trainer.callbacks
                   if type(cb).__name__ == "ModelCheckpoint"), None)
        if mc is None:
            print(f"[CKPT TRACE] Epoch {epoch}: no ModelCheckpoint found!")
            return

        ckpt_dir = mc.dirpath or os.path.join(trainer.log_dir or '.', 'checkpoints')
        print(f"[CKPT TRACE] Epoch {epoch}: post-epoch checkpoint state:")
        print(f"  ModelCheckpoint.last_model_path  = {mc.last_model_path}")
        print(f"  ModelCheckpoint.best_model_path  = {mc.best_model_path}")
        print(f"  ModelCheckpoint.best_model_score = {mc.best_model_score}")

        if os.path.isdir(ckpt_dir):
            for fname, label in [('resume.ckpt', 'resume.ckpt'), ('last.ckpt', 'last.ckpt  ')]:
                fpath = os.path.join(ckpt_dir, fname)
                if os.path.exists(fpath):
                    mb = os.path.getsize(fpath) / 1e6
                    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(fpath))
                    print(f"  {label} on disk: {mb:.1f} MB, mtime {mtime}")
                else:
                    print(f"  {label} DOES NOT EXIST on disk")

            n_ckpts = len([f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')])
            print(f"  total ckpt files   : {n_ckpts}")

        _, _, free = shutil.disk_usage(OUTPUT_FOLDER)
        print(f"  disk free          = {free / 1e9:.2f}GB")

    def on_exception(self, trainer, pl_module, exception):
        print(f"\n[CKPT TRACE] ❌ EXCEPTION: {type(exception).__name__}: {exception}")
        traceback.print_exc()


# ── CLI & Main ────────────────────────────────────────────────────────────────

class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")
        parser.set_defaults({
            "checkpoint.monitor": "eval_loss",
            "checkpoint.save_last": True,
            "checkpoint.save_top_k": 1,
            "checkpoint.filename": "checkpoint-{epoch:02d}-{eval_loss:.2f}-{val_dice:.2f}",
            "checkpoint.every_n_epochs": 1,
        })

    def instantiate_trainer(self, **kwargs):
        run_name = os.environ.get(
            "L2S_RUN_NAME",
            f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        default_root = kwargs.get("default_root_dir", "experiments")
        save_dir = os.path.join(default_root, run_name)
        makedirs(save_dir, exist_ok=True)

        print(f"[System] Initializing Run: {run_name}")
        print(f"[System] All artifacts will be stored in: {save_dir}")

        logger = [
            TensorBoardLogger(save_dir=default_root, name=run_name, version=''),
            CSVLogger(save_dir=default_root, name=run_name, version=''),
        ]

        cbs = kwargs.get("callbacks", []) or []

        # Registration order matters: EveryEpoch writes resume.ckpt first,
        # then CheckpointTrace inspects it, then LossGraph plots.
        cbs.append(EveryEpochCheckpointCallback(filename="resume.ckpt"))
        cbs.append(LossGraphCallback())
        cbs.append(CheckpointTraceCallback())

        print("\n[CLI] Registered callbacks:")
        for i, cb in enumerate(cbs):
            print(f"  [{i}] {type(cb).__name__}")
        print()

        kwargs["default_root_dir"] = save_dir
        kwargs["logger"] = logger
        kwargs["callbacks"] = cbs
        return super().instantiate_trainer(**kwargs)


if __name__ == '__main__':
    cli = CLI(Model, FCDDataModule)
