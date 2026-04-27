import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
import os
from glob import glob
from os import path, makedirs
from ast import literal_eval
from random import shuffle
from typing import Sequence, List, Tuple, Optional, Union
import math
import fnmatch
import random

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import cornucopia as cc
from torch.utils.data import Dataset, DataLoader
from torchmetrics.segmentation import DiceScore as dice_compute

# --- Project Imports ---
# Ensure this points to the correct location of your 'learn2synth' folder
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from learn2synth.networks import UNet, SegNet
from learn2synth.train import SynthSeg
from learn2synth.losses import DiceLoss, LogitMSELoss, CatLoss, CatMSELoss
from learn2synth import optim
from cornucopia import SynthFromLabelTransform, LoadTransform
from learn2synth.utils import folder2files

# --- Configuration ---
# Pointing to your Kaggle dataset structure
DEFAULT_FOLDER = '/kaggle/input/wmh-synthseg-dataset'

class Model(pl.LightningModule):
    def __init__(self,
                 ndim: int = 3,
                 nb_classes: int = 2,  # 0=Background, 1=FCD
                 seg_nb_levels: int = 6,
                 seg_features: Sequence[int] = (16, 32, 64, 128, 256, 512),
                 seg_activation: str = 'ReLU',
                 seg_nb_conv: int = 2,
                 seg_norm: Optional[str] = 'instance',
                 loss: str = 'dice',
                 alpha: float = 1.0,
                 # Parameters for the "Real" logging branch (optional visualization)
                 real_sigma_min: float = 0.15,
                 real_sigma_max: float = 0.15,
                 real_low: float = 0.5,
                 real_middle: float = 0.5,
                 real_high: float = 0.5,
                 classic: bool = True,
                 optimizer: str = 'Adam',
                 optimizer_options: dict = dict(lr=1e-4),
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.optimizer = optimizer
        self.optimizer_options = dict(optimizer_options or {})
        self.alpha = alpha
        
        # Real image augmentation params (for validation visualization only)
        self.real_sigma_min = real_sigma_min
        self.real_sigma_max = real_sigma_max
        self.real_low = real_low
        self.real_middle = real_middle
        self.real_high = real_high
        self.classic = classic

        # --- 1. Segmentation Network (The Student) ---
        segnet = UNet(
            ndim,
            features=seg_features,
            activation=seg_activation,
            nb_levels=seg_nb_levels,
            nb_conv=seg_nb_conv,
            norm=seg_norm,
        )
        segnet = SegNet(ndim, 1, nb_classes, backbone=segnet, activation=None)

        # --- 2. Target Labels ---
        # Note: If you want to force the model to distinguish FCD from normal cortex,
        # consider adding the cortex label here (e.g. `(3,)`) and increasing nb_classes.
        target_labels = [
            (99,),  # Label 1: Focal Cortical Dysplasia (The Legion)
        ]

        # --- 3. Synthesis Generator (The Teacher) ---
        synth = cc.SynthFromLabelTransform(
            target_labels=target_labels,
            one_hot=False,
            elastic=0.05,
            elastic_nodes=10,
            rotation=15,
            shears=0.012,
            zooms=0.15,
            mirror=0.5,
            resolution=5,
            motion_fwhm=2.0,
            gamma=0.5,
            snr=10,
            gmm_fwhm=10,
            bias=7,
            bias_strength=0.5,
        )

        # Wrap in SharedSynth to ensure Real validation images get same geometric warp
        synth = cc.batch(SharedSynth(synth))

        # --- 4. Loss Function ---
        if loss == 'dice':
            # Softmax needed because output is [B, 2, H, W, D]
            loss = DiceLoss(activation='Softmax')
        elif loss == 'logitmse':
            loss = LogitMSELoss()
        elif loss == 'cat':
            loss = CatLoss(activation='Softmax')
        elif loss == 'catmse':
            loss = CatMSELoss(activation='Softmax')
        else:
            raise ValueError('Unsupported loss', loss)

        self.network = SynthSeg(segnet, synth, loss)

        # Manual optimization control (required for Learn2Synth/SynthSeg framework)
        self.automatic_optimization = False
        self.network.set_backward(self.manual_backward)

    def configure_optimizers(self):
        optimizer = getattr(optim, self.optimizer)
        optimizer_init = lambda x: optimizer(x, **(self.optimizer_options or {}))
        optimizers = self.network.configure_optimizers(optimizer_init)
        self.network.set_optimizers(self.optimizers)
        return optimizers

    def training_step(self, batch, batch_idx):
        if self.trainer.current_epoch % 10 == 0 and batch_idx == 0:
            torch.cuda.empty_cache()

        # loss_real is calculated but detached (no gradient)
        loss_synth, loss_real = self.network.synth_and_train_step(*batch)
        
        # Combined for logging purposes only
        loss = loss_synth + self.alpha * loss_real
        
        self.log(f'train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        dice_real = 0

        # Plot/Log images for the first batch of the epoch
        if batch_idx == 0:
            root = f'{self.logger.log_dir}/images'
            makedirs(root, exist_ok=True)
            epoch = self.trainer.current_epoch

            loss_synth, loss_real, pred_synth, pred_real, \
                synth_image, synth_ref, real_image, real_ref \
                = self.network.synth_and_eval_for_plot(*batch)

            # --- Compute Metrics on Real Data ---
            # input_format="index" means inputs are integer class labels, not one-hot
            dice_score = dice_compute(average='micro', include_background=False, 
                                    num_classes=2, input_format="index")

            pred_real = pred_real.cpu()
            real_ref = real_ref.cpu()

            # [B, C, X, Y, Z] -> [B, X, Y, Z] (Integer labels)
            pred_labels = pred_real.argmax(dim=1)
            target_labels = real_ref.squeeze(1)

            dice_score.update(pred_labels, target_labels)
            dice_real = dice_score.compute()

            self.log('dice_real', dice_real, prog_bar=True)

            # Save NIfTI files for visual inspection
            if epoch % 10 == 0:
                pred_synth_argmax = pred_synth[0].argmax(dim=0)
                pred_real_argmax = pred_real[0].argmax(dim=0)
                
                save(pred_synth_argmax, f'{root}/epoch-{epoch:04d}_synth-pred.nii')
                save(pred_real_argmax, f'{root}/epoch-{epoch:04d}_real-pred.nii')
                save(synth_image[0].squeeze(0), f'{root}/epoch-{epoch:04d}_synth-image.nii')
                save(real_image[0].squeeze(0), f'{root}/epoch-{epoch:04d}_real-image.nii')
                # Cast refs to uint8 for label saving
                save(synth_ref[0].squeeze(0).to(torch.uint8), f'{root}/epoch-{epoch:04d}_synth-ref.nii')
                save(real_ref[0].squeeze(0).to(torch.uint8), f'{root}/epoch-{epoch:04d}_real-ref.nii')
        else:
            loss_synth, loss_real = self.network.synth_and_eval_step(*batch)

        loss = loss_synth + self.alpha * loss_real
        self.log('eval_loss', loss)
        return loss

    def forward(self, x):
        return self.network(x)


def save(dat, fname):
    dat = dat.detach().cpu().numpy()
    h = nib.Nifti1Header()
    h.set_data_dtype(dat.dtype)
    nib.save(nib.Nifti1Image(dat, np.eye(4), h), fname)


class SharedSynth(torch.nn.Module):
    """
    Wrapper that applies the exact same geometric deformation to both 
    the Synthetic branch (generation) and the Real branch (augmentation).
    """
    def __init__(self, synth):
        super().__init__()
        self.synth = synth

    def forward(self, slab, img, lab):
        # 1. Sample random parameters based on the label map 'slab'
        final = self.synth.make_final(slab, 1)
        final.deform = final.deform.make_final(slab)
        
        # 2. Generate Synthetic Image
        simg, slab = final(slab)
        
        # 3. Apply SAME deformation to Real Image
        rimg, rlab = final.deform([img, lab])
        rimg = final.intensity(rimg)
        rlab = final.postproc(rlab)
        
        return simg, slab, rimg, rlab


class PairedDataset(Dataset):
    def __init__(self, ndim, images, labels, split_synth_real=True,
                 subset=None, device=None):
        self.ndim = ndim
        self.device = device
        self.split_synth_real = split_synth_real
        self.labels = np.asarray(folder2files(labels)[subset or slice(None)])
        self.images = np.asarray(folder2files(images)[subset or slice(None)])
        
        assert len(self.labels) == len(self.images), \
            "Number of labels and images don't match"

    def __len__(self):
        n = len(self.images)
        if self.split_synth_real:
            n = n // 2
        return n

    def __getitem__(self, idx):
        # Load Real Label and Real Image
        lab = str(self.labels[idx])
        img = str(self.images[idx])

        lab = LoadTransform(ndim=self.ndim, dtype=torch.long, device=self.device)(lab)
        img = LoadTransform(ndim=self.ndim, dtype=torch.float32, device=self.device)(img)

        if self.split_synth_real:
            # If split, use a DIFFERENT label map for synthesis than for validation
            slab = str(self.labels[len(self) + idx])
            slab = LoadTransform(ndim=self.ndim, dtype=torch.long, device=self.device)(slab)
            return slab, img, lab
        else:
            # Shared: Use the SAME label map for synthesis and validation
            return lab, img, lab


class PairedDataModule(pl.LightningDataModule):
    def __init__(self,
                 ndim: int,
                 images: Optional[Sequence[str]] = None,
                 labels: Optional[Sequence[str]] = None,
                 eval: Union[str, slice, List[int], int, float] = 0.2,
                 test: Union[str, slice, List[int], int, float] = 0.2,
                 preshuffle: bool = True,
                 shared: bool = True,
                 batch_size: int = 64,
                 shuffle: bool = False,
                 num_workers: int = 4,
                 prefetch_factor: int = 2,
                 ):
        super().__init__()
        self.ndim = ndim

        # --- DATA LOADING ---
        if labels is None or images is None:
            # Search for sub-XXXX folders
            subject_folders = sorted(glob(path.join(DEFAULT_FOLDER, 'sub-*')))
            self.labels = []
            self.images = []

            print(f"Found {len(subject_folders)} subject folders. Scanning...")

            for subj_dir in subject_folders:
                # Pair FusedMask (Label) with FLAIR (Image)
                label_path = path.join(subj_dir, 'FusedMask.nii')
                image_path = path.join(subj_dir, 'FLAIR.nii')

                if path.exists(label_path) and path.exists(image_path):
                    self.labels.append(label_path)
                    self.images.append(image_path)
            
            print(f"Successfully loaded {len(self.labels)} pairs.")
        else:
            self.labels = list(labels)
            self.images = list(images)

        assert len(self.images) == len(self.labels), "Mismatch in file counts!"

        self.eval = parse_eval(eval)
        self.test = parse_eval(test)
        self.preshuffle = preshuffle
        self.shared = shared
        self.train_kwargs = dict(batch_size=batch_size, shuffle=shuffle, 
                               num_workers=num_workers, prefetch_factor=prefetch_factor)
        self.eval_kwargs = dict(batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers, prefetch_factor=prefetch_factor)

    def setup(self, stage):
        images, labels = self.images, self.labels
        
        if self.preshuffle:
            combined = list(zip(images, labels))
            shuffle(combined)
            images, labels = zip(*combined)
            images, labels = list(images), list(labels)

        # Calculate split indices
        def get_count(param, total):
            if isinstance(param, float): return int(math.ceil(total * param))
            return 0 # Default fallback

        n_eval = get_count(self.eval, len(images))
        n_test = get_count(self.test, len(images))

        # Split: Test -> Eval -> Train
        self.test_images, self.test_labels = images[:n_test], labels[:n_test]
        remaining_images = images[n_test:]
        remaining_labels = labels[n_test:]
        
        self.eval_images, self.eval_labels = remaining_images[:n_eval], remaining_labels[:n_eval]
        self.train_images, self.train_labels = remaining_images[n_eval:], remaining_labels[n_eval:]

    def train_dataloader(self):
        return DataLoader(PairedDataset(self.ndim, self.train_images, self.train_labels, 
                                      split_synth_real=not self.shared), **self.train_kwargs)

    def val_dataloader(self):
        return DataLoader(PairedDataset(self.ndim, self.eval_images, self.eval_labels, 
                                      split_synth_real=not self.shared), **self.eval_kwargs)

    def test_dataloader(self):
        return DataLoader(PairedDataset(self.ndim, self.test_images, self.test_labels, 
                                      split_synth_real=not self.shared), **self.eval_kwargs)

def parse_eval(eval):
    if not isinstance(eval, str): return eval
    try: return literal_eval(eval)
    except: return eval

class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")
        parser.set_defaults({
            "checkpoint.monitor": "eval_loss",
            "checkpoint.save_last": True,
            "checkpoint.save_top_k": 5,
            "checkpoint.every_n_epochs": 10,
        })

if __name__ == '__main__':
    cli = CLI(Model, PairedDataModule)