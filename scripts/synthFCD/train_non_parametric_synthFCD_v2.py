"""FCD SynthSeg – 7-class FCD lesion segmentation (background + 5 tissue groups + FCD lesion)."""
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import sys, os, datetime, glob, random, math
from os import path, makedirs
from ast import literal_eval
from random import shuffle
from typing import Sequence, Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from scipy.ndimage import zoom as scipy_zoom
import cornucopia as cc
from cornucopia import SynthFromLabelTransform, IntensityTransform
from cornucopia.special import IdentityTransform
from monai.transforms import GaussianSmooth
from torch.utils.data import Dataset, DataLoader
from torchmetrics.segmentation import DiceScore as dice_compute

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from learn2synth.networks import UNet, SegNet
from learn2synth.train import SynthSeg
from learn2synth.losses import DiceLoss, LogitMSELoss, CatLoss, CatMSELoss, DiceCELoss, FocalTverskyLoss
from learn2synth import optim
from learn2synth.parameters import FCDParameterCalculator
from learn2synth.augmentations import FCDAugmentations
from learn2synth.custom_cc_synthseg import SynthFromLabelTransform as CustomSynthFromLabelTransform
from cornucopia import IntensityTransform as CustomIntensityTransform

# ── Configuration & subject lists (single source of truth) ──────────────────
from learn2synth.configurations import (
    DEFAULT_FOLDER, CSV_PATH,
    flair_file, roi_file, label_file,
    INTENSITY_SUBJECTS, TRANSMANTLE_SUBJECTS,
    HYPER_SUBJECTS, BLUR_SUBJECTS, THICKENING_SUBJECTS,
)


# ══════════════════════════════════════════════════════════════════════════════
#  SharedSynth  —  geometry + GMM forward pass, intensity kept separate
# ══════════════════════════════════════════════════════════════════════════════

class SharedSynth(torch.nn.Module):
    """
    Applies the same geometric deformation to both the synthetic branch
    (label → image) and the real branch (FLAIR + ROI).

    IntensityTransform is intentionally bypassed here — it is applied
    downstream after FCD augmentations.
    """

    N_CLASSES = 18
    FCD_LESION_LBL = 21
    CORTEX_LBL = 2

    def __init__(self, synth, target_labels=None):
        super().__init__()
        self.synth = synth
        self.target_labels = target_labels or []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, slab, img, lab, roi):
        img = img.float() if img.is_floating_point() else img

        if hasattr(self.synth, 'make_final'):
            return self._forward_standard(slab, img, lab, roi)
        else:
            return self._forward_custom(slab, img, lab, roi)

    # ------------------------------------------------------------------
    # Forward paths
    # ------------------------------------------------------------------

    def _forward_standard(self, slab, img, lab, roi):
        """SynthFromLabelTransform path."""
        final = self.synth.make_final(slab, 1)
        final.deform = final.deform.make_final(slab)

        simg, slab = final(slab)
        rimg, rlab, rroi = final.deform([img, lab, roi])
        rlab = final.postproc(rlab)

        return simg, slab, rimg, rlab, rroi

    def _forward_custom(self, slab, img, lab, roi):
        """
        CustomSynthFromLabelTransform path.

        FCD lesion (label 21) is remapped to cortex (label 2) before GMM
        synthesis so the synthesiser uses the correct tissue prior.
        Lesion appearance is handled downstream by FCD augmentations.
        """
        slab_safe = self._prepare_slab_for_synthesis(slab)
        oh_slab = self._to_one_hot(slab_safe)

        simg, slab_oh, (rimg, rlab, rroi) = self.synth(oh_slab, coreg=[img, lab, roi])

        slab_out = self.remap_labels(slab_oh.argmax(dim=0, keepdim=True))
        rlab = self.remap_labels(rlab)

        return simg, slab_out, rimg, rlab, rroi

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_slab_for_synthesis(self, slab: torch.Tensor) -> torch.Tensor:
        """
        Remap FCD lesion → cortex, then clamp out-of-range values to background.
        """
        slab_safe = slab.clone()
        slab_safe[slab_safe == self.FCD_LESION_LBL] = self.CORTEX_LBL
        slab_safe[(slab_safe < 0) | (slab_safe >= self.N_CLASSES)] = 0
        return slab_safe

    def _to_one_hot(self, label_map: torch.Tensor) -> torch.Tensor:
        """Convert a label map to a one-hot float tensor (C, D, H, W)."""
        return (
            torch.nn.functional.one_hot(
                label_map.long().squeeze(0), num_classes=self.N_CLASSES
            )
            .permute(3, 0, 1, 2)
            .float()
        )

    def remap_labels(self, label_map: torch.Tensor) -> torch.Tensor:
        """Map sparse label values → consecutive class indices (0 = background)."""
        max_value = int(label_map.max().item()) + 1
        lookup_table = torch.zeros(max_value, dtype=torch.long, device=label_map.device)

        for class_index, group in enumerate(self.target_labels, start=1):
            for value in group:
                if value < max_value:
                    lookup_table[value] = class_index

        nb_classes = len(self.target_labels) + 1
        return torch.clamp(lookup_table[label_map.long()], 0, nb_classes - 1)


# ══════════════════════════════════════════════════════════════════════════════
#  Model  —  6-class grouped segmentation (brain structures + FCD lesion)
# ══════════════════════════════════════════════════════════════════════════════

class Model(pl.LightningModule):
    # Class-level label definitions — single source of truth
    TARGET_LABELS = [
        (1,),  # White Matter
        (2,),  # Cerebral Cortex
        (3,),  # Deep Gray Matter
        (4,),  # CSF
        (18,),  # WM — GM Separator
    ]

    def __init__(
            self,
            ndim: int = 3,
            nb_classes: int = 7,  # background + 5 tissues + FCD lesion
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
            csv_path: Optional[str] = CSV_PATH,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.optimizer_name = optimizer
        self.optimizer_options = dict(optimizer_options or {'lr': 1e-4})
        self.time_limit_minutes = time_limit_minutes
        self.alpha = alpha  # weight on the real/FCD-augmented branch
        self.csv_path = csv_path
        self.target_labels = self.TARGET_LABELS

        # Build all sub-modules
        self.subject_params_cache = self._load_subject_params()
        seg_net = self._build_seg_network(ndim, nb_classes, seg_features,
                                          seg_activation, seg_nb_levels,
                                          seg_nb_conv, seg_norm)

        synth = self._build_synth(modality)
        loss_fn = self._build_loss(loss)
        self.network = SynthSeg(seg_net, synth, loss_fn)
        self.intensity_aug = self._build_intensity_aug(modality)

        # Metrics
        _m = dict(include_background=False, num_classes=nb_classes, input_format='index')
        self.val_dice = dice_compute(average='micro', **_m)
        self.val_dice_fcd = dice_compute(average='none', **_m)

        # Manual optimisation
        self.automatic_optimization = False
        self.network.set_backward(self.manual_backward)

    # ══════════════════════════════════════════════════════════════════════
    #  Private builders (called only from __init__)
    # ══════════════════════════════════════════════════════════════════════

    def _load_subject_params(self) -> dict:
        """Pre-load per-subject GMM parameters from CSV into a lookup cache."""
        cache = {}
        if not (self.csv_path and os.path.exists(self.csv_path)):
            return cache
        try:
            import pandas as pd
            from learn2synth.custom_cc_synthseg import FLAIR_CLASS_PARAMS

            df = pd.read_csv(self.csv_path)
            df["subject"] = df["subject"].astype(str).str.strip()
            default_keys = set(range(18)) | set(FLAIR_CLASS_PARAMS.keys())

            for subj in df["subject"].unique():
                params = {
                    int(r["class_id"]): {
                        "mu": (float(r["mu_lo"]), float(r["mu_hi"])),
                        "sigma": (float(r["sigma_lo"]), float(r["sigma_hi"])),
                    }
                    for _, r in df[df["subject"] == subj].iterrows()
                }
                for cls in default_keys:
                    params.setdefault(cls, FLAIR_CLASS_PARAMS.get(cls, {"mu": (0, 255), "sigma": (0, 16)}))
                cache[subj] = params

            print(f"[Model] Preloaded per-subject params for {len(cache)} subjects.")
        except Exception as exc:
            print(f"[Model] Warning: failed to parse CSV — {exc}")
        return cache

    def _build_seg_network(self, ndim, nb_classes, features, activation,
                           nb_levels, nb_conv, norm):
        backbone = UNet(ndim, features=features, activation=activation,
                        nb_levels=nb_levels, nb_conv=nb_conv, norm=norm)
        return SegNet(ndim, 1, nb_classes, backbone=backbone, activation=None)

    def _build_synth(self, modality: str) -> SharedSynth:
        if modality == 'flair':
            from learn2synth.custom_cc_synthseg import FLAIR_CLASS_PARAMS
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
            'dice': lambda: DiceLoss(activation='Softmax'),
            'logitmse': lambda: LogitMSELoss(),
            'cat': lambda: CatLoss(activation='Softmax'),
            'catmse': lambda: CatMSELoss(activation='Softmax'),
            'dice_ce': lambda: DiceCELoss(weighted=False, lambda_dice=1.0,
                                          lambda_ce=1.0, activation='Softmax'),
            'focal_tversky': lambda: FocalTverskyLoss(activation='Softmax'),
        }
        if loss not in options:
            raise ValueError(f"Unsupported loss '{loss}'. Choose from: {list(options)}")
        return options[loss]()

    def _build_intensity_aug(self, modality: str):
        if modality == 'flair':
            return CustomIntensityTransform(
                bias=5, gamma=0.4, motion_fwhm=2, resolution=4, snr=15, gfactor=3, order=3,
            )
        return IntensityTransform(
            bias=7, bias_strength=0.2, gamma=0.3, motion_fwhm=3,
            resolution=4, snr=20, gfactor=2, order=3,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  Label utilities
    # ══════════════════════════════════════════════════════════════════════

    def remap_labels(self, label_map: torch.Tensor) -> torch.Tensor:
        """
        Remap sparse FreeSurfer label values → consecutive model class indices.
        Anything not in target_labels becomes 0 (background).
        """
        max_val = int(label_map.max().item()) + 1
        lut = torch.zeros(max_val, dtype=torch.long, device=label_map.device)
        for cls_idx, group in enumerate(self.target_labels, start=1):
            for val in group:
                if val < max_val:
                    lut[val] = cls_idx
        nb_classes = len(self.target_labels) + 1
        return torch.clamp(lut[label_map.long()], 0, nb_classes - 1)

    def _set_subject_params(self, subject_id: Optional[str]):
        """Swap GMM class_params for the current subject (or fall back to global defaults)."""
        gmm = getattr(getattr(self.network.synth, 'synth', None), 'gmm', None)
        if gmm is None or not hasattr(gmm, 'class_params'):
            return
        if subject_id and subject_id in self.subject_params_cache:
            gmm.class_params = self.subject_params_cache[subject_id]
        else:
            from learn2synth.custom_cc_synthseg import FLAIR_CLASS_PARAMS
            gmm.class_params = FLAIR_CLASS_PARAMS

    # ══════════════════════════════════════════════════════════════════════
    #  Augmentation pipeline
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _parse_aug_choices(aug_type: str) -> list:
        if aug_type == 'combo':
            return random.sample(['blur', 'zoom', 'hyper', 'trans'], random.randint(1, 4))
        return aug_type.split('+')

    def _apply_fcd_augmentations(self, img: torch.Tensor, roi: torch.Tensor,
                                 choices: list, params: dict) -> torch.Tensor:
        """Apply the requested FCD augmentation chain in-place (returns new tensor)."""
        aug = FCDAugmentations()
        for ch in choices:
            if ch == 'blur':
                img = aug.apply_roi_augmentations_blured(
                    img, roi, sigma_range=params['blur_sigma'])
            elif ch == 'zoom':
                img = aug.apply_roi_thickening(
                    img, roi, zoom_range=params['zoom_f'])
            elif ch in ('hyper', 'trans'):
                img = aug.apply_roi_augmentations_hyperintensity(
                    img, roi,
                    intensity_range=params['int_rng'],
                    sigma_range=params['hyper_sigma'] if ch == 'hyper' else params['trans_sigma'],
                )
        return img

    def _maybe_save_aug_sample(self, aug_img: torch.Tensor, aug_mask: torch.Tensor,
                               aug_type: str):
        """Save one NIfTI sample per aug_type for debugging (fires once per type)."""
        if not hasattr(self, '_saved_aug_types'):
            self._saved_aug_types = set()
        if aug_type in self._saved_aug_types:
            return
        self._saved_aug_types.add(aug_type)
        import nibabel as nib, numpy as np
        save_dir = '/kaggle/working/saved_augs'
        os.makedirs(save_dir, exist_ok=True)
        nib.save(nib.Nifti1Image(aug_img.detach().cpu().numpy(), np.eye(4)),
                 os.path.join(save_dir, f'subj_aug_{aug_type}_img.nii.gz'))
        nib.save(nib.Nifti1Image(aug_mask.detach().cpu().numpy().astype(np.uint8), np.eye(4)),
                 os.path.join(save_dir, f'subj_aug_{aug_type}_mask.nii.gz'))

    def _process_single_sample(self, batch: dict, i: int):
        """
        Full single-sample pipeline:
          SharedSynth → FCD aug → IntensityTransform → label fusion.
        Returns (aug_image, aug_mask, real_image, real_mask) or None if skipped.
        """
        label_t = batch['label_t'][i]
        flair_t = batch['flair_t'][i].float()
        roi_t = batch['roi_t'][i]
        aug_type = batch['aug_type'][i]
        subject_id = batch.get('subject_id', [None] * len(batch['label_t']))[i]

        aug_params = {
            'int_rng': (batch['int_factor_min'][i].item(), batch['int_factor_max'][i].item()),
            'blur_sigma': (batch['blur_sigma_min'][i].item(), batch['blur_sigma_max'][i].item()),
            'zoom_f': (batch['zoom_f_min'][i].item(), batch['zoom_f_max'][i].item()),
            'hyper_sigma': (batch['hyper_sigma_min'][i].item(), batch['hyper_sigma_max'][i].item()),
            'trans_sigma': (batch['trans_sigma_min'][i].item(), batch['trans_sigma_max'][i].item()),
        }

        if label_t.sum() == 0 or torch.isnan(label_t.float()).any():
            return None

        self._set_subject_params(subject_id)

        # Step 1: SharedSynth (geometry + GMM, no intensity)
        simg, slab, rimg, rlab, rroi = self.network.synth(label_t, flair_t, label_t, roi_t)
        simg_3d = simg.squeeze(0).float()
        slab_3d = slab.squeeze(0).long()
        rroi_3d = (rroi.squeeze(0) > 0).long()

        # Step 2: FCD augmentations
        choices = self._parse_aug_choices(aug_type)
        aug_img = self._apply_fcd_augmentations(simg_3d.clone(), rroi_3d, choices, aug_params)

        # Step 3: Debug save (once per aug_type)
        self._maybe_save_aug_sample(aug_img, rroi_3d, aug_type)

        # Step 4: Standalone IntensityTransform
        aug_out = self.intensity_aug(aug_img.float().unsqueeze(0))
        aug_image_item = aug_out[0] if isinstance(aug_out, (list, tuple)) else aug_out

        # Step 5: Fuse FCD lesion label (class 6) into segmentation maps
        slab_with_fcd = slab_3d.clone()
        slab_with_fcd[rroi_3d > 0] = 6

        rlab_with_fcd = rlab.long().squeeze(0).clone()
        rlab_with_fcd[rroi_3d > 0] = 6

        return (
            aug_image_item,  # (1, Z, Y, X)
            slab_with_fcd.unsqueeze(0),  # (1, Z, Y, X)
            rimg.float(),
            rlab_with_fcd.unsqueeze(0),
        )

    def synthesize_batch(self, batch: dict):
        """
        Run the full augmentation pipeline for every sample in the batch.
        Returns (aug_image, aug_mask, real_image, real_mask) stacked tensors.
        """
        results = []
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            for i in range(len(batch['label_t'])):
                out = self._process_single_sample(batch, i)
                if out is not None:
                    results.append(out)

        aug_images, aug_masks, real_images, real_masks = zip(*results)
        return (
            torch.stack(aug_images).float(),
            torch.stack(aug_masks).long(),
            torch.stack(real_images).float(),
            torch.stack(real_masks).long(),
        )

    # ══════════════════════════════════════════════════════════════════════
    #  Training / Validation
    # ══════════════════════════════════════════════════════════════════════

    def _forward_both_branches(self, aug_image, aug_mask, real_image, real_mask):
        """Shared forward + loss computation for both branches."""
        pred_synth = self.network.segnet(aug_image)
        loss_synth = self.network.loss(pred_synth, aug_mask)

        real_image = real_image.to(self.device)
        real_labels = real_mask.squeeze(1).to(self.device)
        pred_real = self.network.segnet(real_image)
        loss_real = self.network.loss(pred_real, real_labels.unsqueeze(1))

        return pred_synth, loss_synth, pred_real, real_labels, loss_real

    # def training_step(self, batch, batch_idx):
    #     if self.trainer.current_epoch % 10 == 0 and batch_idx == 0:
    #         torch.cuda.empty_cache()
    #
    #     aug_image, aug_mask, real_image, real_mask = self.synthesize_batch(batch)
    #
    #     opt = self.optimizers()
    #     opt.zero_grad()
    #     self.train()
    #
    #     _, loss_synth, _, _, loss_real = self._forward_both_branches(
    #         aug_image, aug_mask, real_image, real_mask)
    #
    #     loss = loss_synth + self.alpha * loss_real
    #     self.manual_backward(loss)
    #     opt.step()
    #     self.log('train_loss', loss, prog_bar=True)
    #     return loss

    def training_step(self, batch, batch_idx):
        if self.trainer.current_epoch % 10 == 0 and batch_idx == 0:
            torch.cuda.empty_cache()

        aug_image, aug_mask, real_image, real_mask = self.synthesize_batch(batch)

        # Store the optimizer getter method so train_step() can access it
        self.network.optimizers = self.optimizers

        # Now call train_step which will use the stored optimizer reference
        loss_synth, loss_real = self.network.train_step(
            aug_image, aug_mask, real_image, real_mask)

        self.log('train_loss_synth', loss_synth, prog_bar=True)
        self.log('train_loss_real', loss_real, prog_bar=False)
        return loss_synth

    def validation_step(self, batch, batch_idx):
        self.eval()
        with torch.no_grad():
            aug_image, aug_mask, real_image, real_mask = self.synthesize_batch(batch)
            pred_synth, loss_synth, pred_real, real_labels, loss_real = self._forward_both_branches(
                aug_image, aug_mask, real_image, real_mask)

        pred_labels = pred_real.argmax(dim=1)
        self.val_dice.update(pred_labels.cpu(), real_labels.cpu())
        self.val_dice_fcd.update(pred_labels.cpu(), real_labels.cpu())

        if batch_idx == 0:
            self._log_val_diagnostics(pred_synth, pred_labels, aug_image,
                                      real_image, aug_mask, real_labels)

        loss = loss_synth + self.alpha * loss_real
        self.log('eval_loss', loss, prog_bar=True)
        return loss

    def _log_val_diagnostics(self, pred_synth, pred_labels, aug_image,
                             real_image, aug_mask, real_labels):
        """Log class-count scalars and save NIfTI samples every 10 epochs."""
        pred_synth_argmax = pred_synth[0].argmax(dim=0)
        pred_real_argmax = pred_labels[0]
        self.log('pred_synth_num_classes', float(len(torch.unique(pred_synth_argmax))), prog_bar=False)
        self.log('pred_real_num_classes', float(len(torch.unique(pred_real_argmax))), prog_bar=False)

        epoch = self.trainer.current_epoch
        if epoch % 10 != 0:
            return

        base_dir = self.trainer.log_dir or self.trainer.default_root_dir
        img_root = os.path.join(base_dir, 'images')
        makedirs(img_root, exist_ok=True)
        print(f'\n[Saving] Sample images for Epoch {epoch} to {img_root}...')

        p = f'{img_root}/epoch-{epoch:04d}'
        save(pred_synth_argmax, f'{p}_synth-pred.nii.gz')
        save(pred_real_argmax, f'{p}_real-pred.nii.gz')
        save(aug_image[0].squeeze(0), f'{p}_synth-image.nii.gz')
        save(real_image[0].squeeze(0), f'{p}_real-image.nii.gz')
        save(aug_mask[0].squeeze(0).to(torch.uint8), f'{p}_synth-ref.nii.gz')
        save(real_labels[0].to(torch.uint8), f'{p}_real-ref.nii.gz')

    def on_validation_epoch_end(self):
        dice_epoch = self.val_dice.compute()
        dice_per_cls = self.val_dice_fcd.compute()
        dice_fcd = dice_per_cls[5] if len(dice_per_cls) > 5 else torch.tensor(0.0)

        self.log('val_dice', dice_epoch, prog_bar=True)
        self.log('val_dice_fcd', dice_fcd, prog_bar=False)

        tl = self.trainer.callback_metrics.get('train_loss', -1)
        el = self.trainer.callback_metrics.get('eval_loss', -1)
        print(f"\n{'=' * 40}")
        print(f"EPOCH {self.trainer.current_epoch} SUMMARY:")
        print(f"  Train Loss    : {tl:.4f}")
        print(f"  Eval Loss     : {el:.4f}")
        print(f"  DICE SCORE    : {dice_epoch:.4f}")
        print(f"  DICE FCD (c6) : {dice_fcd:.4f}")
        print(f"{'=' * 40}\n")

        self.val_dice.reset()
        self.val_dice_fcd.reset()

        el_metric = self.trainer.callback_metrics.get('eval_loss')
        if el_metric is not None and hasattr(self, '_scheduler'):
            loss_val = el_metric.item() if hasattr(el_metric, 'item') else float(el_metric)
            self._scheduler.step(loss_val)
            current_lr = self._scheduler.optimizer.param_groups[0]['lr']
            print(f'  LR            : {current_lr:.2e}')

    # ══════════════════════════════════════════════════════════════════════
    #  Optimiser / callbacks / inference
    # ══════════════════════════════════════════════════════════════════════

    def configure_optimizers(self):
        opt_cls = getattr(optim, self.optimizer_name)
        optimizer = opt_cls(self.network.segnet.parameters(), **(self.optimizer_options or {}))
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-6,
        )
        return optimizer

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
            import traceback;
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
            import traceback;
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
        import shutil
        _, _, free = shutil.disk_usage('/kaggle/working')
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

        import shutil
        _, _, free = shutil.disk_usage('/kaggle/working')
        print(f"  disk free          = {free / 1e9:.2f}GB")

    def on_exception(self, trainer, pl_module, exception):
        print(f"\n[CKPT TRACE] ❌ EXCEPTION: {type(exception).__name__}: {exception}")
        import traceback;
        traceback.print_exc()


# ── FCDDataset ────────────────────────────────────────────────────────────────
# Returns un-augmented volumes plus random augmentation configurations.
# Actual GPU synthesis happens inside Model.synthesize_batch

class FCDDataset(Dataset):
    def __init__(self, ndim, label_paths, flair_paths, roi_paths, fcd_intensity_range=(0.1, 0.5), fcd_tail_length_range=(20, 50)):
        self.ndim = ndim
        self.fcd_intensity_range = fcd_intensity_range
        self.fcd_tail_length_range = fcd_tail_length_range

        self.items = []
        for label_path, flair_path, roi_path in zip(label_paths, flair_paths, roi_paths):
            subject_num = FCDParameterCalculator().get_subj_num(os.path.dirname(label_path))
            aug_matches = []
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

        # --- Load volumes ---
        flair_arr = nib.load(flair_path).get_fdata()
        label_arr = nib.load(label_path).get_fdata().astype(int)
        roi_arr = nib.load(roi_path).get_fdata().astype(int)

        # --- Resample to FLAIR space if shapes mismatch ---
        if flair_arr.shape != roi_arr.shape:
            roi_arr = (FCDParameterCalculator().resample_to_target(roi_arr, flair_arr.shape, True) > 0.5).astype(int)
        if flair_arr.shape != label_arr.shape:
            label_arr = FCDParameterCalculator().resample_to_target(label_arr, flair_arr.shape, True).astype(int)

        # --- Convert to tensors ---
        # ROI is passed through SharedSynth as fused slab; FCD lesion recovered as class 7 after postprocessing.
        label_tensor = torch.as_tensor(label_arr, dtype=torch.int64).unsqueeze(0)
        flair_tensor = torch.as_tensor(flair_arr, dtype=torch.float32).unsqueeze(0)
        roi_tensor = torch.as_tensor(roi_arr, dtype=torch.int64).unsqueeze(0)

        # --- Augmentation parameter ranges (pre-sampled on CPU for clean worker RNG) ---
        aug_params = {
            'int_factor_min': torch.tensor(self.fcd_intensity_range[0], dtype=torch.float32),
            'int_factor_max': torch.tensor(self.fcd_intensity_range[1], dtype=torch.float32),
            'tail_length_min': torch.tensor(self.fcd_tail_length_range[0], dtype=torch.long),
            'tail_length_max': torch.tensor(self.fcd_tail_length_range[1], dtype=torch.long),
            'blur_sigma_min': torch.tensor(0.7, dtype=torch.float32),
            'blur_sigma_max': torch.tensor(1.7, dtype=torch.float32),
            'zoom_f_min': torch.tensor(0.2, dtype=torch.float32),
            'zoom_f_max': torch.tensor(0.4, dtype=torch.float32),
            'hyper_sigma_min': torch.tensor(0.0, dtype=torch.float32),
            'hyper_sigma_max': torch.tensor(0.3, dtype=torch.float32),
            'trans_sigma_min': torch.tensor(0.0, dtype=torch.float32),
            'trans_sigma_max': torch.tensor(0.3, dtype=torch.float32),
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

class FCDDataModule(pl.LightningDataModule):
    def __init__(self,
                 ndim: int = 3,
                 dataset_path: str = DEFAULT_FOLDER,
                 eval: float = 0.04,
                 preshuffle: bool = False,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 num_workers: int = 4):
        super().__init__()
        self.ndim = ndim
        self.dataset_path = dataset_path
        self.eval_frac = eval
        self.preshuffle = preshuffle
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        # --- Scan for valid (label, flair, roi) triplets ---
        subject_folders = sorted(glob.glob(path.join(dataset_path, 'sub-*')))
        lp_all, fp_all, rp_all = [], [], []
        for sd in subject_folders:
            lp = path.join(sd, label_file)
            fp = path.join(sd, flair_file)
            rp = path.join(sd, roi_file)
            if all(path.exists(x) for x in [lp, fp, rp]):
                lp_all.append(lp)
                fp_all.append(fp)
                rp_all.append(rp)

        n_subj = len(lp_all)
        n_folders = len(subject_folders)

        assert n_subj > 0, (
            f"[FCDDataModule] Fatal: found 0 valid triplets in {dataset_path}. "
            "Check your file names and folder structure."
        )

        if n_subj < n_folders:
            print(
                f"[FCDDataModule] WARNING: {n_subj}/{n_folders} subjects have complete triplets "
                f"— {n_folders - n_subj} dropped due to missing files."
            )
        else:
            print(f"[FCDDataModule] {n_subj} subjects loaded (perfect match).")

        self.label_paths = lp_all
        self.flair_paths = fp_all
        self.roi_paths = rp_all

        # --- Compute FCD augmentation parameters ---
        print("[FCDDataModule] Computing FCD augmentation parameters…")
        self.fcd_int_rng, self.fcd_tl_rng = FCDParameterCalculator().calculate_fcd_parameters(
            dataset_path=dataset_path,
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

        lp, fp, rp = self.label_paths, self.flair_paths, self.roi_paths
        if self.preshuffle:
            combined = list(zip(lp, fp, rp))
            shuffle(combined)
            lp, fp, rp = map(list, zip(*combined))

        n = len(lp)

        def _count(param, total):
            if isinstance(param, float): return int(math.ceil(total * param))
            if isinstance(param, int):   return param
            return 0

        # Clean 2-way split (Train / Val)
        ne = _count(self.eval_frac, n)
        eval_lp, eval_fp, eval_rp = lp[:ne], fp[:ne], rp[:ne]
        train_lp, train_fp, train_rp = lp[ne:], fp[ne:], rp[ne:]

        kw = dict(
            fcd_intensity_range=self.fcd_int_rng,
            fcd_tail_length_range=self.fcd_tl_rng,
        )

        self.train_ds = FCDDataset(self.ndim, train_lp, train_fp, train_rp, **kw)
        self.eval_ds = FCDDataset(self.ndim, eval_lp, eval_fp, eval_rp, **kw)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=True if self.num_workers > 0 else False)

    def val_dataloader(self):
        return DataLoader(self.eval_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=True if self.num_workers > 0 else False)


# ── CLI & Main ────────────────────────────────────────────────────────────────

def parse_eval(val):
    if not isinstance(val, str):
        return val
    try:
        return literal_eval(val)
    except Exception:
        return val


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
