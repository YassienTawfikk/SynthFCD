import os
import glob
import math
from random import shuffle
from os import path
from ast import literal_eval
from typing import Union, List, Sequence, Optional

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from cornucopia import SynthFromLabelTransform, LoadTransform

# --- Custom learn2synth imports ---
from learn2synth.parameters import FCDParameterCalculator
from learn2synth.configurations import (
    DEFAULT_FOLDER,
    BLUR_SUBJECTS,
    THICKENING_SUBJECTS,
    HYPER_SUBJECTS,
    TRANSMANTLE_SUBJECTS,
    INTENSITY_SUBJECTS,
    label_file,
    flair_file,
    roi_file,
    train_file,
    input_file
)
from learn2synth.utils import folder2files


def parse_eval(eval):
    if not isinstance(eval, str): return eval
    try:
        return literal_eval(eval)
    except:
        return eval


# ── FCDDataset ────────────────────────────────────────────────────────────────
# Returns un-augmented volumes plus random augmentation configurations.
# Actual GPU synthesis happens inside Model.synthesize_batch

class FCDDataset(Dataset):
    AUG_TYPES = ['blur', 'zoom', 'hyper', 'trans', 'combo']

    def __init__(self, ndim, label_paths, flair_paths, roi_paths,
                 fcd_intensity_range=(0.1, 0.5),
                 fcd_tail_length_range=(20, 50)):
        self.ndim = ndim
        self.int_rng = fcd_intensity_range
        self.tl_rng = fcd_tail_length_range

        self.items = []
        for lp, fp, rp in zip(label_paths, flair_paths, roi_paths):
            sn = FCDParameterCalculator().get_subj_num(os.path.dirname(lp))
            added = False
            matches = []
            if sn in BLUR_SUBJECTS: matches.append('blur')
            if sn in THICKENING_SUBJECTS: matches.append('zoom')
            if sn in HYPER_SUBJECTS: matches.append('hyper')
            if sn in TRANSMANTLE_SUBJECTS: matches.append('trans')

            if not matches:
                self.items.append((lp, fp, rp, 'combo'))
            else:
                self.items.append((lp, fp, rp, '+'.join(matches)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        lp, fp, rp, aug_type = self.items[idx]

        fl_d = nib.load(fp).get_fdata()
        la_d = nib.load(lp).get_fdata().astype(int)
        ro_d = nib.load(rp).get_fdata().astype(int)

        if fl_d.shape != ro_d.shape:
            ro_d = (FCDParameterCalculator().resample_to_target(ro_d, fl_d.shape, True) > 0.5).astype(int)
        if fl_d.shape != la_d.shape:
            la_d = FCDParameterCalculator().resample_to_target(la_d, fl_d.shape, True).astype(int)

        # ROI is no longer passed separately: SharedSynth deforms fusedmask as slab,
        # and FCD lesion is recovered as class 7 (slab_3d == 7) after postprocessing.
        label_t = torch.as_tensor(la_d, dtype=torch.int64).unsqueeze(0)
        flair_t = torch.as_tensor(fl_d, dtype=torch.float32).unsqueeze(0)
        roi_t = torch.as_tensor(ro_d, dtype=torch.int64).unsqueeze(0)

        # Pre-generate random augmentation parameters on the CPU 
        # so workers handle RNG cleanly without state issues.
        return {
            'label_t': label_t,
            'flair_t': flair_t,
            'roi_t': roi_t,
            'aug_type': aug_type,
            'int_factor_min': torch.tensor(self.int_rng[0], dtype=torch.float32),
            'int_factor_max': torch.tensor(self.int_rng[1], dtype=torch.float32),

            # Pass ranges down for random sampling inside GPU training step if needed, 
            # or pre-sample them completely. We will pre-sample min/max as scalar bounds 
            # and just pass them.
            'tail_length_min': torch.tensor(self.tl_rng[0], dtype=torch.long),
            'tail_length_max': torch.tensor(self.tl_rng[1], dtype=torch.long),
            'blur_sigma_min': torch.tensor(0.7, dtype=torch.float32),
            'blur_sigma_max': torch.tensor(1.7, dtype=torch.float32),
            'zoom_f_min': torch.tensor(0.2, dtype=torch.float32),
            'zoom_f_max': torch.tensor(0.4, dtype=torch.float32),
            'hyper_sigma_min': torch.tensor(0.0, dtype=torch.float32),
            'hyper_sigma_max': torch.tensor(0.3, dtype=torch.float32),
            'trans_sigma_min': torch.tensor(0.0, dtype=torch.float32),
            'trans_sigma_max': torch.tensor(0.3, dtype=torch.float32),
        }


# ── FCDDataModule ─────────────────────────────────────────────────────────────

class FCDDataModule(pl.LightningDataModule):
    def __init__(self,
                 ndim: int = 3,
                 dataset_path: str = DEFAULT_FOLDER,
                 eval: float = 0.2,
                 test: float = 0.0,
                 preshuffle: bool = True,
                 batch_size: int = 1,
                 num_workers: int = 4):  # INCREASED FROM 0
        super().__init__()
        self.ndim = ndim
        self.dataset_path = dataset_path
        self.eval_frac = eval
        self.test_frac = test
        self.preshuffle = preshuffle
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Scan for fusedmask.nii (= synthseg.nii + roi.nii) paired with flair.nii
        subject_folders = sorted(glob.glob(path.join(dataset_path, 'sub-*')))
        lp_all, fp_all, rp_all = [], [], []
        for sd in subject_folders:
            lp = path.join(sd, label_file)  # fusedmask.nii
            fp = path.join(sd, flair_file)
            rp = path.join(sd, roi_file)
            # l = path.join(sd, label_file)
            if all(path.exists(x) for x in [lp, fp, rp]):
                lp_all.append(lp);
                fp_all.append(fp);
                rp_all.append(rp)
        self.label_paths = lp_all
        self.flair_paths = fp_all
        self.roi_paths = rp_all
        n_subj = len(lp_all)
        print(f"[FCDDataModule] {n_subj} subjects found.")

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

        nt = _count(self.test_frac, n)
        ne = _count(self.eval_frac, n)

        test_lp, test_fp, test_rp = lp[:nt], fp[:nt], rp[:nt]
        rem_lp, rem_fp, rem_rp = lp[nt:], fp[nt:], rp[nt:]
        eval_lp, eval_fp, eval_rp = rem_lp[:ne], rem_fp[:ne], rem_rp[:ne]
        train_lp, train_fp, train_rp = rem_lp[ne:], rem_fp[ne:], rem_rp[ne:]

        kw = dict(fcd_intensity_range=self.fcd_int_rng, fcd_tail_length_range=self.fcd_tl_rng)
        self.train_ds = FCDDataset(self.ndim, train_lp, train_fp, train_rp, **kw)
        self.eval_ds = FCDDataset(self.ndim, eval_lp, eval_fp, eval_rp, **kw)
        self.test_ds = FCDDataset(self.ndim, test_lp, test_fp, test_rp, **kw)

    def _dl(self, ds, shuffle_it):
        # By setting num_workers we speed up the bare loading
        return DataLoader(ds, batch_size=self.batch_size,
                          shuffle=shuffle_it,
                          num_workers=self.num_workers,
                          pin_memory=True)  # CHANGED TO TRUE FOR GPU DATALOADING

    def train_dataloader(self):
        return self._dl(self.train_ds, shuffle_it=True)

    def val_dataloader(self):
        return self._dl(self.eval_ds, shuffle_it=False)

    def test_dataloader(self):
        return self._dl(self.test_ds, shuffle_it=False)


class PairedDataset(Dataset):
    def __init__(self, ndim, images, labels, split_synth_real=True,
                 subset=None, device=None):
        self.ndim = ndim
        self.device = device
        self.split_synth_real = split_synth_real
        self.labels = np.asarray(folder2files(labels)[subset or slice(None)])
        self.images = np.asarray(folder2files(images)[subset or slice(None)])

        assert len(self.labels) == len(self.images), "Number of labels and images don't match"

    def __len__(self):
        n = len(self.images)
        if self.split_synth_real:
            n = n // 2
        return n

    def __getitem__(self, idx):
        lab = str(self.labels[idx])
        img = str(self.images[idx])

        lab = LoadTransform(ndim=self.ndim, dtype=torch.long, device=self.device)(lab)
        img = LoadTransform(ndim=self.ndim, dtype=torch.float32, device=self.device)(img)

        if self.split_synth_real:
            slab = str(self.labels[len(self) + idx])
            slab = LoadTransform(ndim=self.ndim, dtype=torch.long, device=self.device)(slab)
            return slab, img, lab
        else:
            return lab, img, lab


class PairedDataModule(pl.LightningDataModule):
    def __init__(self,
                 ndim: int,
                 images: Optional[Sequence[str]] = None,
                 labels: Optional[Sequence[str]] = None,
                 eval: Union[str, slice, List[int], int, float] = 0.04,
                 preshuffle: bool = False,
                 shared: bool = True,
                 batch_size: int = 64,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 prefetch_factor: int = 2,
                 ):
        super().__init__()
        self.ndim = ndim

        if labels is None or images is None:
            subject_folders = sorted(glob(path.join(DEFAULT_FOLDER, 'sub-*')))
            self.labels = []
            self.images = []

            print(f"Found {len(subject_folders)} subject folders. Scanning...")

            for subj_dir in subject_folders:
                label_path = path.join(subj_dir, f'{train_file}')
                image_path = path.join(subj_dir, f'{input_file}')

                if path.exists(label_path) and path.exists(image_path):
                    self.labels.append(label_path)
                    self.images.append(image_path)

            print(f"Successfully loaded {len(self.labels)} pairs ")
            print(f"{len(subject_folders) - len(self.labels)} skipped).")
        else:
            self.labels = list(labels)
            self.images = list(images)

        assert len(self.images) == len(self.labels), "Mismatch in file counts!"

        self.eval = parse_eval(eval)
        self.preshuffle = preshuffle
        self.shared = shared
        self.train_kwargs = dict(batch_size=batch_size, shuffle=shuffle,
                                 num_workers=num_workers, prefetch_factor=prefetch_factor)
        self.eval_kwargs = dict(batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, prefetch_factor=prefetch_factor)

    def setup(self, stage):
        if hasattr(self, '_setup_done'):
            return

        self._setup_done = True
        images, labels = self.images, self.labels

        if self.preshuffle:
            combined = list(zip(images, labels))
            shuffle(combined)
            images, labels = zip(*combined)
            images, labels = list(images), list(labels)

        def get_count(param, total):
            if isinstance(param, float):
                return int(math.ceil(total * param))
            if isinstance(param, int):
                return param
            return 0

        n_eval = get_count(self.eval, len(images))

        self.eval_images, self.eval_labels = images[:n_eval], labels[:n_eval]
        self.train_images, self.train_labels = images[n_eval:], labels[n_eval:]

        print(f"Split — Train: {len(self.train_images)}, Val: {len(self.eval_images)}")

    def train_dataloader(self):
        return DataLoader(PairedDataset(self.ndim, self.train_images, self.train_labels,
                                        split_synth_real=not self.shared), **self.train_kwargs)

    def val_dataloader(self):
        return DataLoader(PairedDataset(self.ndim, self.eval_images, self.eval_labels,
                                        split_synth_real=not self.shared), **self.eval_kwargs)
