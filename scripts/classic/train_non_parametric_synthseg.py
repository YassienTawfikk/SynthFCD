import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # or hardcode if needed
sys.path.append(project_root)

from learn2synth.networks import UNet, SegNet
from learn2synth.train import SynthSeg
from learn2synth.losses import DiceLoss, LogitMSELoss, CatLoss, CatMSELoss
from learn2synth import optim
from cornucopia import (
    SynthFromLabelTransform, LoadTransform, NonFinalTransform, FinalTransform
)
import cornucopia as cc
from learn2synth.utils import folder2files
from torch.utils.data import Dataset, DataLoader
from typing import Sequence, List, Tuple, Optional, Union
from glob import glob
from os import path, makedirs
from ast import literal_eval
from random import shuffle
import nibabel as nib
import numpy as np
import torch
import math
import fnmatch
from torchmetrics.segmentation import DiceScore as dice_compute

default_folder = '/kaggle/input/wmh-synthseg-dataset'


class Model(pl.LightningModule):
    def __init__(self,
                 ndim: int = 3,
                 nb_classes: int = 2,
                 seg_nb_levels: int = 6,
                 seg_features: Sequence[int] = (16, 32, 64, 128, 256, 512),
                 seg_activation: str = 'ReLU',
                 seg_nb_conv: int = 2,
                 seg_norm: Optional[str] = 'instance',
                 synth_nb_levels: int = 1,
                 synth_features: Tuple[int] = (16,),
                 synth_activation: str = 'ELU',
                 synth_nb_conv: int = 1,
                 synth_norm: Optional[str] = None,
                 synth_residual: bool = True,
                 #  synth_shared: bool = True,
                 loss: str = 'dice',
                 alpha: float = 1.,
                 real_sigma_min: float = 0.15,
                 real_sigma_max: float = 0.15,
                 real_low: float = 0.5,
                 real_middle: float = 0.5,
                 real_high: float = 0.5,
                 classic: bool = True,
                 optimizer: str = 'Adam',
                 optimizer_options: dict = dict(lr=1e-4),
                 # metrics: dict = dict(dice='dice'),
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.optimizer = optimizer
        self.optimizer_options = dict(optimizer_options or {})
        self.alpha = alpha
        self.real_sigma_min = real_sigma_min
        self.real_sigma_max = real_sigma_max
        self.real_low = real_low
        self.real_middle = real_middle
        self.real_high = real_high
        self.classic = classic

        segnet = UNet(
            ndim,
            features=seg_features,
            activation=seg_activation,
            nb_levels=seg_nb_levels,
            nb_conv=seg_nb_conv,
            norm=seg_norm,
        )
        segnet = SegNet(ndim, 1, nb_classes, backbone=segnet, activation=None)

        target_labels = [
            (99,),  # Focal Cortical Dysplasia
            # (2,),   # Left Cerebral White Matter
            # (3,),   # Left Cerebral Cortex
            # (4,),   # Left Lateral Ventricle
            # (5,),   # Left Inferior Lateral Ventricle
            # (7,),   # Left Cerebellum White Matter
            # (8,),   # Left Cerebellum Cortex
            # (10,),  # Left Thalamus
            # (11,),  # Left Caudate
            # (12,),  # Left Putamen
            # (13,),  # Left Pallidum
            # (14,),  # 3rd Ventricle
            # (15,),  # 4th Ventricle
            # (16,),  # Brain Stem
            # (17,),  # Left Hippocampus
            # (18,),  # Left Amygdala
            # (24,),  # CSF
            # (26,),  # Left Accumbens Area
            # (28,),  # Left Ventral DC
            # (41,),  # Right Cerebral White Matter
            # (42,),  # Right Cerebral Cortex
            # (43,),  # Right Lateral Ventricle
            # (44,),  # Right Inferior Lateral Ventricle
            # (46,),  # Right Cerebellum White Matter
            # (47,),  # Right Cerebellum Cortex
            # (49,),  # Right Thalamus
            # (50,),  # Right Caudate
            # (51,),  # Right Putamen
            # (52,),  # Right Pallidum
            # (53,),  # Right Hippocampus
            # (54,),  # Right Amygdala
            # (58,),  # Right Accumbens Area
            # (60,),  # Right Ventral DC
            # (77,),  # WMH (White Matter Hyperintensities)
        ]

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
        synth = cc.batch(SharedSynth(synth))

        if loss == 'dice':
            # Weight classes to handle severe imbalance (background: 99.7%, hippocampus: 0.3%)
            # Background gets low weight, hippocampus classes get high weight
            # class_weights = [0.1, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
            # loss = DiceLoss(activation=None, weighted=class_weights)
            loss = DiceLoss(activation='Softmax')
        elif loss == 'logitmse':
            loss = LogitMSELoss()
        elif loss == 'cat':
            loss = CatLoss(activation='Softmax')
        elif loss == 'catmse':
            loss = CatMSELoss(activation='Softmax')
        elif isinstance(loss, str):
            raise ValueError('Unsupported loss', loss)

        self.network = SynthSeg(segnet, synth, loss)

        self.automatic_optimization = False
        self.network.set_backward(self.manual_backward)

    def configure_optimizers(self):
        optimizer = getattr(optim, self.optimizer)
        optimizer_init = lambda x: optimizer(x, **(self.optimizer_options or {}))
        optimizers = self.network.configure_optimizers(optimizer_init)
        self.network.set_optimizers(self.optimizers)
        return optimizers

    def training_step(self, batch, batch_idx):
        # self.trainer.fit_loop.max_epochs = 10000        
        if self.trainer.current_epoch % 10 == 0 and batch_idx == 0:
            torch.cuda.empty_cache()

        loss_synth, loss_real = self.network.synth_and_train_step(*batch)
        loss = loss_synth + self.alpha * loss_real
        name = type(self.network.loss).__name__
        # self.log(f'train_loss_synth_{name}', loss_synth)
        # self.log(f'train_loss_real_{name}', loss_real)
        # self.log(f'train_loss_{name}', loss)

        self.log(f'train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        dice_real = 0

        if batch_idx == 0:
            root = f'{self.logger.log_dir}/images'
            makedirs(root, exist_ok=True)
            epoch = self.trainer.current_epoch

            loss_synth, loss_real, pred_synth, pred_real, \
                synth_image, synth_ref, real_image, real_ref \
                = self.network.synth_and_eval_for_plot(*batch)

            # Initialize Dice metric (once per step)
            dice_score = dice_compute(average='micro', include_background=False, num_classes=2, input_format="index")

            # Move to CPU and convert predictions to label indices
            pred_real = pred_real.cpu()  # [B, C, H, W, D]
            real_ref = real_ref.cpu()  # [B, 1, H, W, D]

            # Convert logits/probs to predicted labels
            pred_labels = pred_real.argmax(dim=1)  # [B, H, W, D]
            target_labels = real_ref.squeeze(1)  # [B, H, W, D]

            # Update metric with batch tensors (no loop)
            dice_score.update(pred_labels, target_labels)

            # Compute Dice
            dice_real = dice_score.compute()

            # Log the computed Dice score
            self.log('dice_real', dice_real, prog_bar=True)

            # Save predicted label maps for first batch element
            pred_synth_argmax = pred_synth[0].argmax(dim=0)
            pred_real_argmax = pred_real[0].argmax(dim=0)

            self.log('pred_synth_num_classes', len(torch.unique(pred_synth_argmax)), prog_bar=True)
            self.log('pred_real_num_classes', len(torch.unique(pred_real_argmax)), prog_bar=True)

            if epoch % 10 == 0:
                save(pred_synth_argmax, f'{root}/epoch-{epoch:04d}_synth-pred.nii')
                save(pred_real_argmax, f'{root}/epoch-{epoch:04d}_real-pred.nii')
                save(synth_image[0].squeeze(0), f'{root}/epoch-{epoch:04d}_synth-image.nii')
                save(real_image[0].squeeze(0), f'{root}/epoch-{epoch:04d}_real-image.nii')
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
    # dat = dat[:, :, None]
    dat = dat.detach().cpu().numpy()
    h = nib.Nifti1Header()
    h.set_data_dtype(dat.dtype)
    nib.save(nib.Nifti1Image(dat, np.eye(4), h), fname)


class SharedSynth(torch.nn.Module):
    """Apply the same geometric transform for synth and real"""

    def __init__(self, synth):
        super().__init__()
        self.synth = synth

    def forward(self, slab, img, lab):
        final = self.synth.make_final(slab, 1)
        final.deform = final.deform.make_final(slab)
        simg, slab = final(slab)
        rimg, rlab = final.deform([img, lab])
        rimg = final.intensity(rimg)
        rlab = final.postproc(rlab)
        return simg, slab, rimg, rlab


class PairedDataset(Dataset):
    """
    A dataset of paired images and labels.

    The dataset returns a three-tuple of:
        1) a label map to use for synth
        2) a real image
        3) a real label map
    If `split_synth_real` is False, the same label map is used for (1)
    and (3). Otherwise, the dataset is split in two, and the first half
    is used for synthesis and the other half for real examples.
    """

    def __init__(self, ndim, images, labels, split_synth_real=True,
                 subset=None, device=None):
        """

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        labels : [sequence of] str
            Input label maps: folder or file pattern or list of files.
        images : [sequence of] str
            Input images: folder or file pattern or list of files.
        split_synth_real : bool
            Do not use the same label map for real and synth examples
            in each batch.
        subset : slice or list[int]
            Only use a subset of the input files
        device : torch.device
            Device on which to load the data
        """
        self.ndim = ndim
        self.device = device
        self.split_synth_real = split_synth_real
        self.labels = np.asarray(folder2files(labels)[subset or slice(None)])
        self.images = np.asarray(folder2files(images)[subset or slice(None)])
        # NOTE: array[char] use less RAM that list[str] in multithreads
        assert len(self.labels) == len(self.images), \
            "Number of labels and images don't match"

    def __len__(self):
        n = len(self.images)
        if self.split_synth_real:
            n = n // 2
        return n

    def __getitem__(self, idx):
        lab, img = str(self.labels[idx]), str(self.images[idx])

        lab = LoadTransform(ndim=self.ndim, dtype=torch.long, device=self.device)(lab)
        img = LoadTransform(ndim=self.ndim, dtype=torch.float32, device=self.device)(img)

        # mean = 0
        # var = 10
        # sigma = var ** 0.5
        # gaussian = torch.from_numpy(np.random.normal(mean, sigma, (img.shape[0], img.shape[1], img.shape[2])))
        # gaussian = gaussian.float()
        # img = img + gaussian

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
                 eval: Union[str, slice, List[int], int, float] = 0.2,
                 test: Union[str, slice, List[int], int, float] = 0.2,
                 preshuffle: bool = True,
                 shared: bool = False,
                 batch_size: int = 64,
                 shuffle: bool = False,
                 num_workers: int = 4,
                 prefetch_factor: int = 2,
                 ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        images : sequence[str]
            List of images. By default, uses `orig_talairach_slice` from
            vxmdata1, in folders that have `samseg_23_talairach_slice` labels.
        labels : sequence[str]
            List of label maps. By default, uses `samseg_23_talairach_slice`
            from vxmdata1.
        eval : float
            Percentage of images to keep for evaluation
        test : float
            Percentage of images to keep for test
        preshuffle : bool
            Shuffle the image order once (otherwise, sort alphabetically)
        shared : bool
            Use the same image for synth and real loss in each minibatch,
            otherwise use different images.
        batch_size : int
            Number of elements in a minibatch
        shuffle : bool
            Shuffle file order at each epoch
        num_workers : int
            Number of workers for the dataloader
        prefetch_factor : int
            Number of batches to pre-load in advance
        """
        super().__init__()
        self.ndim = ndim

        # --- NEW DATA LOADING LOGIC ---
        if labels is None or images is None:
            # Find all subject directories starting with 'sub-'
            subject_folders = sorted(glob(path.join(default_folder, 'sub-*')))

            self.labels = []
            self.images = []

            print(f"Found {len(subject_folders)} subject folders. Searching for files...")

            for subj_dir in subject_folders:
                # Expected structure based on your dataset:
                # Label map: FusedMask.nii
                # Real image: FLAIR.nii
                label_path = path.join(subj_dir, 'FusedMask.nii')
                image_path = path.join(subj_dir, 'FLAIR.nii')

                # Only include pairs where both files exist
                if path.exists(label_path) and path.exists(image_path):
                    self.labels.append(label_path)
                    self.images.append(image_path)
                else:
                    # Uncomment to log skipped subjects
                    # print(f"Skipping {path.basename(subj_dir)} (missing FusedMask.nii or FLAIR.nii)")
                    pass

            print(f"Loaded {len(self.labels)} image-label pairs.")

        else:
            # Use manually provided file lists
            self.labels = list(labels)
            self.images = list(images)

        # Safety check: must have same count
        assert len(self.images) == len(self.labels), "Mismatch between image and label list lengths!"

        # --- REST OF ORIGINAL LOGIC ---
        self.eval = parse_eval(eval)
        self.test = parse_eval(test)
        self.preshuffle = preshuffle
        self.shared = shared

        self.train_kwargs = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        self.eval_kwargs = dict(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )

    def setup(self, stage):
        images = self.images
        labels = self.labels
        if self.preshuffle:
            image_and_labels = list(zip(images, labels))
            shuffle(image_and_labels)
            images = [i for i, _ in image_and_labels]
            labels = [l for _, l in image_and_labels]

        slice_eval = self.eval
        if isinstance(slice_eval, float):
            slice_eval = int(math.ceil(len(images) * slice_eval))
        slice_test = self.test
        if isinstance(slice_test, float):
            slice_test = int(math.ceil(len(images) * slice_test))
        remaining_images, remaining_labels, \
            self.test_images, self.test_labels \
            = splitset(images, labels, slice_test)
        self.train_images, self.train_labels, \
            self.eval_images, self.eval_labels \
            = splitset(remaining_images, remaining_labels, slice_eval)

    def train_dataloader(self):
        train_dataset = PairedDataset(
            self.ndim, self.train_images, self.train_labels,
            split_synth_real=not self.shared)
        return DataLoader(train_dataset, **self.train_kwargs)

    def val_dataloader(self):
        eval_dataset = PairedDataset(
            self.ndim, self.eval_images, self.eval_labels,
            split_synth_real=not self.shared)
        return DataLoader(eval_dataset, **self.eval_kwargs)

    def test_dataloader(self):
        test_dataset = PairedDataset(
            self.ndim, self.test_images, self.test_labels,
            split_synth_real=not self.shared)
        return DataLoader(test_dataset, **self.eval_kwargs)


def parse_eval(eval):
    if not isinstance(eval, str):
        return eval
    if ':' in eval:
        eval = eval.split(':')
        eval = map(literal_eval, eval)
        eval = slice(*eval)
    else:
        try:
            eval = literal_eval(eval)
        except ValueError:
            pass
    return eval


def splitset(images, labels, eval):
    if isinstance(eval, float):
        eval = int(math.ceil(len(images) * eval))
    if isinstance(eval, int):
        eval = slice(-eval, None)
    if isinstance(eval, (slice, list, tuple)):
        eval_images = images[eval]
        eval_labels = labels[eval]
    else:
        eval_images = list(sorted(fnmatch.filter(images, eval)))
        eval_labels = list(sorted(fnmatch.filter(labels, eval)))
    images = list(sorted(filter(lambda x: x not in eval_images, images)))
    labels = list(sorted(filter(lambda x: x not in eval_labels, labels)))
    return images, labels, eval_images, eval_labels


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
