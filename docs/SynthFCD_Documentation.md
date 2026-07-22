# SynthFCD — Technical Documentation

Complete technical record of the SynthFCD training pipeline: architecture, design decisions,
configuration, diagnostics, known pitfalls, and change history.

This document is the authoritative reference for anyone extending or reproducing this work.
It assumes familiarity with the [README](../README.md) and with SynthSeg-style domain
randomization in general.

**Sections most likely to save you time:**
[Things to Be Careful With](#8-things-to-be-careful-with) ·
[Configuration Reference](#5-configuration-reference) ·
[Open Issues](#10-open-issues)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [The Three Approaches](#3-the-three-approaches)
4. [Key Design Decisions](#4-key-design-decisions)
5. [Configuration Reference](#5-configuration-reference)
6. [Evaluation & Diagnostics](#6-evaluation--diagnostics)
7. [Failure-Mode Analysis](#7-failure-mode-analysis)
8. [Things to Be Careful With](#8-things-to-be-careful-with)
9. [Change History](#9-change-history)
10. [Open Issues](#10-open-issues)

---

## 1. Project Overview

### Task

7-class voxel segmentation of FLAIR MRI volumes:

| Class | Tissue | Source label |
| :---: | :--- | :--- |
| 0 | Background | 0 |
| 1 | White Matter | 1 |
| 2 | Cerebral Cortex | 2 |
| 3 | Deep Gray Matter | 3 |
| 4 | CSF | 4 |
| 5 | WM–GM Separator | 18 |
| 6 | **FCD Lesion** | 21, or the binary ROI mask |

### Label Map Encoding

Source label maps use an 18-class SuperSynth encoding:

| Labels | Content |
| :--- | :--- |
| 0–4 | Core brain structures (background, WM, cortex, deep GM, CSF) |
| 5–17 | Extra-cerebral structures (optic chiasm, air, artery, skin, bone, …) |
| 18 | WM–GM boundary → model class 5 |
| 21 | FCD lesion — **present only in `fusedmask.nii`**, never in `labelmap.nii` |

Labels 5–17 participate in synthesis (they generate realistic skull, scalp, and background
texture) but collapse to background in the output label map.

### Dataset

Bonn FCD Type II cohort, [OpenNeuro ds004199](https://openneuro.org/datasets/ds004199).

| Pool | Count | Use |
| :--- | :---: | :--- |
| `train/raw/` | 57 | Training + validation split |
| `train/generated/` | 80 | Additional training subjects (label-space augmented) |
| Test | 28 | Held-out evaluation |
| Healthy controls | 5 | Available, not currently used in training |

Per-subject files:

```
sub-XXXXX/
├── flair.nii        # real FLAIR volume
├── t1.nii           # co-registered T1
├── labelmap.nii     # 18-class SuperSynth anatomical labels
├── roi.nii          # binary FCD lesion mask
└── fusedmask.nii    # labelmap + ROI fused, lesion = label 21
```

Subjects are categorized by radiological phenotype in `learn2synth/configurations.py`.
A subject may appear in multiple lists; subjects in none are assigned `'combo'`.

```python
TRANSMANTLE_SUBJECTS = [24, 33, 38, 60, 65, 78, 83, 87, 101, 109, 116, 123]
HYPER_SUBJECTS       = [1, 3, 14, 15, 16, 18, 27, 40, 50, 53, 55, 58, 63, 73, 77,
                        89, 97, 98, 112, 122, 130, 133, 138, 146]
BLUR_SUBJECTS        = [44, 55, 65, 80, 81, 105, 115, 126, 132]
THICKENING_SUBJECTS  = [10, 18, 24, 43, 47, 58, 59, 63, 68, 71, 72, 76, 89, 90,
                        91, 97, 116, 120, 122, 131, 133, 138, 139, 140, 141, 146]
INTENSITY_SUBJECTS   = HYPER_SUBJECTS + TRANSMANTLE_SUBJECTS
```

### Hardware

Developed on Kaggle. P100 (single GPU) is the primary target; T4 was explored for DDP and
rolled back. **P100 has compute capability 6.0 and is incompatible with PyTorch ≥ 2.5** —
pin `torch ≤ 2.4` or switch to T4.

---

## 2. Architecture

### Network

```yaml
type: SegNet (3D UNet)
nb_features: [16, 32, 64, 128, 256, 512]
nb_levels: 6
nb_classes: 7
```

Checkpoint state-dict keys are prefixed `network.segnet.` — relevant when loading weights
outside Lightning.

### Component Map

```
train_non_parametric_synthFCD.py
├── FLAIR_CLASS_PARAMS (module-level global)
│     loaded from FLAIR_CLASS_PARAMS_CSV at import time
│     global fallback GMM prior for all subjects
│
├── FCDDataset (torch.utils.data.Dataset)
│     packs volume tensors + augmentation parameter ranges per subject
│     loads fusedmask_t when approach='nativeSynth'
│     assigns aug_type per subject from the phenotype lists
│
├── FCDDataModule (pl.LightningDataModule)
│     source-aware split: raw/ for val, raw/ + generated/ for train
│     seeded deterministic split via split_seed
│     computes fcd_intensity_range + fcd_tail_range from real subjects only
│
├── SharedSynth (torch.nn.Module)
│     wraps CustomSynthFromLabelTransform  (flair_modality=True)
│           or cornucopia SynthFromLabelTransform (flair_modality=False)
│     routes on hasattr(self.synth, 'make_final')
│     owns: _to_one_hot, remap_labels
│
├── SynthesisPipelineDebugger
│     saves intermediate NIfTI volumes at every synthesis stage
│
├── Model (pl.LightningModule)
│     owns: SynthSeg, SharedSynth, FCDAugmentations, IntensityTransform
│     subject_params_cache: per-subject GMM ranges from FLAIR_STATS_CSV
│     synthesis: _process_single_sample → synthesize_batch
│     training: manual optimization (automatic_optimization=False)
│
└── CLI (LightningCLI subclass)
      loggers, callbacks, run directory
      link_arguments: model.approach → data.approach
```

### `learn2synth` Package

| File | Role |
| :--- | :--- |
| `custom_cc_synthseg.py` | FLAIR GMM transforms: `SynthFromLabelTransform`, `PerClassGaussianMixtureTransform`, `load_class_params_from_csv` |
| `networks.py` | `UNet` backbone + `SegNet` wrapper |
| `modules.py` | `ConvBlockBase` and conv/norm/activation building blocks |
| `train.py` | `SynthSeg` — owns `train_step` and `eval_for_plot` |
| `augmentations.py` | `FCDAugmentations` — FCD appearance injection |
| `parameters.py` | `FCDParameterCalculator` — dataset-level statistics |
| `losses.py` | `DiceLoss`, `DiceCELoss`, `CatLoss`, `FocalTverskyLoss`, `CatMSELoss`, `LogitMSELoss` |
| `configurations.py` | All path constants and subject lists |

### Attribute Chain

```
Model.network                                  → SynthSeg
SynthSeg.synth                                 → SharedSynth
SharedSynth.synth                              → CustomSynthFromLabelTransform  (flair_modality=True)
                                               → SynthFromLabelTransform        (flair_modality=False)
CustomSynthFromLabelTransform.gmm              → PerClassGaussianMixtureTransform
PerClassGaussianMixtureTransform.class_params  → swapped per step by _set_subject_params
```

`_set_subject_params` calls `self.network.synth.set_class_params(params)`, which reaches
`self.synth.gmm.class_params` internally.

### Callbacks

| Callback | Purpose |
| :--- | :--- |
| `TimeLimitCallback` | Stops training after N minutes; sets `should_stop` on all ranks to avoid DDP deadlock |
| `LossGraphCallback` | Writes `training_plot.png` each epoch from `metrics.csv` + `metrics_backup.csv` |
| `CheckpointTraceCallback` | Diagnostics — disk usage, checkpoint state |
| `ModelCheckpoint` ×4 | `eval_loss` (min), `val_dice` (max), `val_dice_fcd` (max), and one unmonitored writing `last.ckpt` every epoch |

---

## 3. The Three Approaches

Selected via `--model.approach`. The value propagates to the data module automatically
through `link_arguments` — **never set `--data.approach` separately.**

| Value | Meaning |
| :--- | :--- |
| `synthFCD` | Post-hoc lesion appearance injection via `FCDAugmentations` |
| `nativeSynth` | Lesion as a first-class GMM label |
| `normal` | Direct supervision on real FLAIR — no synthesis |

`native_synthesis` survives internally as a derived flag (`approach == 'nativeSynth'`) but
is not a CLI argument.

### 3.1 `synthFCD` — Post-Hoc Injection

```
DISK: flair.nii + labelmap.nii + roi.nii
        │
        ▼ FCDDataset.__getitem__
        Load NIfTIs → tensors + augmentation parameter ranges
        │
        ▼ Model._process_single_sample
        1. validate: input_t = label_t
        2. _set_subject_params()      → swap GMM to per-subject FLAIR statistics
        3. SharedSynth (deform + GMM) → synthetic image + warped labels + warped ROI
        4. _apply_fcd_augmentations() → inject FCD appearance (hyper / blur / zoom / trans)
        5. IntensityTransform         → bias field, gamma, noise, resolution → [0, 1]
        6. Label fusion               → roi voxels → class 6 in both label maps
        │
        ▼ SynthSeg.train_step  [GPU]
        SYNTHETIC BRANCH: segnet(aug_image)  → loss → backward
        REAL BRANCH:      segnet(real_image) → loss → contributes via α
```

The lesion region is labeled **cortex** in the input label map, so the GMM gives it
normal cortex intensities. `FCDAugmentations` then modifies those intensities inside the
ROI, and label fusion marks the region as class 6 in the target.

### 3.2 `nativeSynth` — Lesion as a GMM Label

```
DISK: fusedmask.nii + flair.nii
        │
        ▼ FCDDataset.__getitem__
        Load NIfTIs → tensors (fusedmask_t has labels {0..18, 21})
        │
        ▼ Model._process_single_sample
        1. validate: input_t = fusedmask_t
        2. _set_subject_params()      → if flair_modality: inject class 21 GMM params
        3. SharedSynth (deform + GMM) → slab = lab = fusedmask_t, roi = None
           • label 21 gets its own GMM channel → unique intensities
           • remap_labels: 21 → class 6 in both output label maps
        4. IntensityTransform         → bias field, gamma, noise, resolution → [0, 1]
        │
        ▼ SynthSeg.train_step  [GPU]
```

No `FCDAugmentations`, no label fusion, no ROI passed to `SharedSynth` — lesion location
is encoded in the label map itself.

### 3.3 `normal` — Real-Data Control

Direct supervision on augmented real FLAIR. Runs its **own** single-branch
forward/backward in `training_step` and never calls `SynthSeg.train_step`. Changes to the
two-branch loop do not affect it.

> `use_extra_data` must be `false` for this approach — generated subjects carry no real
> lesion annotation. A warning is printed if both are set.

---

## 4. Key Design Decisions

### 4.1 Two-CSV GMM System

| CSV | Content | Role |
| :--- | :--- | :--- |
| `FLAIR_CLASS_PARAMS_CSV` | Global per-class (μ, σ) ranges | Module-level `FLAIR_CLASS_PARAMS`, fallback for all subjects |
| `FLAIR_STATS_CSV` | Per-subject per-class (μ, σ) ranges | `subject_params_cache` — tightened prior per subject |

Before each forward pass, `_set_subject_params(subject_id)` swaps the GMM's `class_params`
to that subject's version, falling back to the global prior for unseen subjects. On the
`nativeSynth` + FLAIR path, class 21 parameters are injected through a **shallow copy** so
the cache is never mutated.

### 4.2 `SharedSynth` Routing

`SharedSynth.forward` routes on `hasattr(self.synth, 'make_final')`:

- **FLAIR path** (`flair_modality=True`) → `CustomSynthFromLabelTransform`, a plain
  `nn.Module` with no `make_final`. Deformation + per-class GMM. `IntensityTransform`
  disabled internally (`no_augs=True`), applied downstream.
- **Standard path** (`flair_modality=False`) → cornucopia's `SynthFromLabelTransform`,
  which has `make_final`. Full deformation + shared GMM. Intensity disabled via
  `raw.intensity = IdentityTransform()`, applied downstream.

`flair_modality=False` runs roughly 20% slower per epoch — `make_final` has per-sample
instantiation overhead and cornucopia's deformation pipeline is heavier. This is expected,
not a bug.

### 4.3 `no_augs=True` Semantics

`no_augs=True` disables **only** `self.intensity` → `donothing`. Geometric deformation
always runs. When `no_augs=True`, `QuantileTransform(clip=True)` is applied after GMM
output so `simg` exits in `[0, 1]` regardless of path.

### 4.4 One-Hot Encoding and the GMM

Labels 0–18 → one-hot `(19, D, H, W)`. The GMM samples one (μ, σ) per channel.
On `nativeSynth`: labels 0–18 + 21 → one-hot `(22, D, H, W)`, giving label 21 a dedicated
channel and therefore unique lesion intensities.

`num_classes` must be `N_CLASSES + 1` = 19 (standard) or 22 (native). Passing 18 crashes.

### 4.5 Why Label 21 Crashes Cornucopia

Cornucopia's internal LUT is sized to `max(target_labels) + 1 = 19` (labels 0–18).
Indexing it at 21 is out of bounds. Even without the crash, cornucopia's `postproc` maps
unlisted labels to background — it has no concept of 21 → class 6.

Both problems are solved by pre-remapping 21 → 19 before cornucopia on the standard path:

- avoids the crash
- gives the lesion its own GMM channel
- `rlab` (deformed fusedmask from `final.deform`) retains label 21, so `remap_labels`
  still produces class 6 correctly

### 4.6 Frozen Deformation

A fresh random field applied independently to image and label map de-registers them.
The field is frozen once and re-applied:

```python
frozen_deform = self.deform.make_final(x)       # freeze to x's spatial shape

img_out = frozen_deform(x)                       # cubic — intensities
frozen_deform.nearest_if_label = True
lab_out = frozen_deform(lab.round().long())      # nearest-neighbour — labels
```

`make_final` is also what guarantees consistent output shapes across subjects. Without it,
`RandomAffineElasticTransform` produced variable shapes from different input geometries and
crashed the UNet decoder.

### 4.7 Normalization

`QuantileTransform(clip=True)` maps the 1st–99th percentile of each volume to `[0, 1]`,
excluding background zeros. It replaces the earlier hardcoded `/255.0`.

- `/255.0` assumes a fixed maximum. A subject peaking at 180 uses only 70% of the range;
  one peaking at 300 has real signal clipped.
- `clip=True` is mandatory — the default `clip=False` leaves the lower tail negative, and
  negative values entering `RandomGammaTransform` (`x^gamma`) produce NaN.
- The `make_final(x)(x)` pattern is required: `QuantileTransform` is a `NonFinalTransform`
  and cannot be called directly.

Applied in two places: to `simg` inside `SynthFromLabelTransform` when `no_augs=True`, and
to `rimg` in `_process_single_sample` before `IntensityTransform`.

### 4.8 Two-Branch Training

```python
def train_step(self, synth_image, synth_ref, real_image, real_ref):
    synth_pred = self.segnet(synth_image)
    synth_loss = self.loss(synth_pred, synth_ref)

    real_pred  = self.segnet(real_image)
    real_loss  = self.loss(real_pred, real_ref)

    total = synth_loss + self.alpha * real_loss
    self.backward(total)
    optim.step()

    return synth_loss, real_loss
```

> **Revision-dependent.** Earlier revisions ran the real branch under `eval()` +
> `no_grad()` as a monitoring-only diagnostic, logging `real_loss` without
> backpropagating it. The real branch was later found to be silently contributing
> gradients, and the two-branch form above was adopted deliberately, with
> `batch_size` reduced to 1 to avoid OOM. **Check `learn2synth/train.py` for the state of
> the revision you are reproducing** — results from before and after this change are not
> directly comparable.

`automatic_optimization = False` plus `self.network.set_backward(self.manual_backward)` is
what makes the two-branch step expressible at all; Lightning's automatic loop cannot
represent it. `manual_backward` routes through Lightning's AMP gradient scaler.

### 4.9 Augmentation Type Assignment

Each subject receives a fixed `aug_type` at `FCDDataset.__init__` from its membership in
the phenotype lists. Multiple memberships combine with `+` (e.g. `zoom-hyper`). Subjects in
no list get `'combo'` — 1–4 randomly chosen strategies per step.

On `nativeSynth`, `aug_type` is still assigned but never used.

### 4.10 Pre-Sampled Augmentation Scalars

`FCDAugmentations` methods accept **pre-sampled scalars**, not range tuples. The caller
samples every random value upfront into a `pre` dict:

| Old (range-based) | New (scalar) |
| :--- | :--- |
| `zoom_range=params['zoom_f']` | `zoom_factor=pre['zoom_factor']` |
| `intensity_range=params['int_rng']` | `intensity_factor=pre['hyper_intensity']` |
| `sigma_range=params['blur_sigma']` | `sigma=pre['blur_sigma']` |

This guarantees each subject gets a statistically independent draw once per step regardless
of the order in which augmentations are composed.

### 4.11 `clean_val`

Validation can run on augmented real FLAIR (`clean_val=False`, default) or clean
un-deformed real FLAIR (`clean_val=True`).

`_clean_real_pair()` applies `QuantileTransform(clip=True)` only — no deformation, no
`IntensityTransform` — then remaps labels. Only the **real** branch swaps; the synthetic
branch is unchanged. Training always uses `clean=False`.

Clean validation matches deployment conditions, reduces variance, and gives a more reliable
signal to `ReduceLROnPlateau` and to checkpoint selection. It also produces higher, smoother
numbers — **switching resets the baseline and breaks comparability with earlier epochs.**

---

## 5. Configuration Reference

### Paths

```python
# learn2synth/configurations.py
DEFAULT_FOLDER         = '.../augmented-fcd-lesion-mri-dataset/fcd/'
FLAIR_STATS_CSV        = '.../flair_stats_raw_fcd_train.csv'
FLAIR_CLASS_PARAMS_CSV = '.../flair_class_params_fcd_train.csv'

flair_file     = 'flair.nii'
roi_file       = 'roi.nii'
label_file     = 'labelmap.nii'
fusedmask_file = 'fusedmask.nii'
```

### Model Parameters

```yaml
model:
  approach: synthFCD           # synthFCD | nativeSynth | normal
  flair_modality: false        # true = custom FLAIR GMM path
  clean_val: false
  lesion_gmm_params: null      # default {mu: (100, 180), sigma: (5, 20)}; native+flair only
  loss: dice_ce                # dice_ce | focal_tversky
  optimizer: Adam
  optimizer_options: {lr: 0.0001}
  seg_features: [16, 32, 64, 128, 256, 512]
  seg_nb_levels: 6
  nb_classes: 7
  alpha: 1.0                   # real-branch weight
  n_tracked_batches: 2
  val_diagnostics_interval: 10
  debug_subject_ids: null
data:
  eval: 0.04                   # fraction of raw subjects held out
  batch_size: 1
  num_workers: 4
  use_extra_data: false
  split_seed: 42
  fcd_intensity_range: null    # supply to skip the 57-folder scan
  fcd_tail_range: null
```

### Environment Variables

| Variable | Effect |
| :--- | :--- |
| `L2S_TIME_LIMIT_MINUTES` | Overrides `time_limit_minutes` at train start |
| `L2S_RUN_NAME` | Sets the experiment directory name |

### Launch Commands

**SynthFCD:**
```bash
python train_non_parametric_synthFCD.py fit \
  --model.approach synthFCD \
  --model.flair_modality true \
  --data.dataset_path data/fcd/ \
  --data.use_extra_data true \
  --data.split_seed 42 \
  --data.fcd_intensity_range "[0.02, 0.3602]" \
  --data.fcd_tail_range "[14, 29]" \
  --trainer.max_epochs 2000 \
  --trainer.accelerator gpu \
  --trainer.num_sanity_val_steps 0 \
  --seed_everything 0
```

**Native SynthSeg:**
```bash
python train_non_parametric_synthFCD.py fit \
  --model.approach nativeSynth \
  --model.flair_modality true \
  --data.dataset_path data/fcd/ \
  --data.use_extra_data true \
  --trainer.max_epochs 2000 --trainer.accelerator gpu
```

**Normal (control):**
```bash
python train_non_parametric_synthFCD.py fit \
  --model.approach normal \
  --model.flair_modality true \
  --data.dataset_path data/fcd/ \
  --data.use_extra_data false \
  --trainer.max_epochs 2000 --trainer.accelerator gpu
```

### Resuming

```bash
--ckpt_path experiments/<run>/checkpoints/last.ckpt
```

`last.ckpt` is written every epoch by a **dedicated unmonitored** `ModelCheckpoint`
(`save_top_k=0, save_last=True, every_n_epochs=1`). See
[§8](#8-things-to-be-careful-with) for why this matters.

Verify a checkpoint is not frozen:

```python
import torch, os
p = "experiments/<run>/checkpoints/last.ckpt"
print("islink:", os.path.islink(p), "->", os.path.realpath(p))
print("epoch :", torch.load(p, map_location="cpu", weights_only=False)["epoch"])
```

### Checkpoint Patching

`save_hyperparameters()` bakes hyperparameters into the checkpoint. Adding or renaming a
`Model.__init__` parameter breaks resume from older checkpoints. Patch the stored dict and
write to a writable path:

```python
import torch, pathlib

src = "experiments/<run>/checkpoints/last.ckpt"
dst = pathlib.Path("/kaggle/working/patched_ckpts/last.ckpt")
dst.parent.mkdir(parents=True, exist_ok=True)

ck = torch.load(src, map_location="cpu", weights_only=False)
hp = ck["hyper_parameters"]

for stale in ("n_best_batches", "native_synthesis", "_instantiator"):
    hp.pop(stale, None)

hp["loss"] = "focal_tversky"          # rewrite as needed

torch.save(ck, dst)
```

Alternatively, use `save_hyperparameters(ignore=[...])` for volatile parameters such as
`prob_map_subject_ids` so they never enter the checkpoint.

### Dependencies

- `pytorch-lightning`
- `cornucopia` — pinned to `6f8ab58dfcfe8978c9aa9e8b05898dcf7d75bb5b`
- `torchmetrics`, `nibabel`, `scipy`
- `torch ≤ 2.4` on P100

---

## 6. Evaluation & Diagnostics

### Per-Subject Validation Metrics

Every `val_diagnostics_interval` epochs, `validation_step` computes and logs per-subject
Dice and loss under `val_dice_<subject_id>` / `val_loss_<subject_id>`. A fresh
`dice_compute` instance is created per subject to prevent state bleed.

### Best / Worst Batch Tracking

`_val_batch_cache` and `_val_worst_cache` maintain the top-N and bottom-N validation
batches by loss. At epoch end, NIfTI diagnostics are written for both:

```
images/epoch-0480/best_sub-00005/real-image.nii.gz
images/epoch-0480/worst_sub-00033/real-image.nii.gz
```

### Subject Metrics CSV

`subject_metrics.csv` in wide format — one row per epoch, one column pair per subject:

```
epoch, sub-00001 (dice), sub-00001 (fcd_dice), sub-00003 (dice), ...
```

Appends across runs, preserving full history.

### Epoch Summary

```
════════════════════════════════════════
 epoch 480  |  val_dice 0.4412  |  lr 1.00×10⁻⁴
 ── Subject Diagnostics ────────────────
 best  sub-00097  dice=0.4706  fcd=0.0314
 worst sub-00033  dice=0.2860  fcd=0.0002
════════════════════════════════════════
```

### Synthesis Pipeline Debugger

```bash
--model.debug_subject_ids '["sub-00001", "sub-00033", "sub-00044"]'
```

Writes NIfTI volumes for every synthesis stage, once per subject per run:

| Stage | Content |
| :---: | :--- |
| 0 | Raw inputs |
| 1 | Post-synthesis (deform + GMM) |
| 2 | Post-FCD-augmentation |
| 3 | Post-`IntensityTransform` |
| 4 | Post-label-fusion |
| 5 | `rimg` after normalization + `IntensityTransform` |

Output: `pipeline_debug_{modality_tag}_{synth_tag}/{subject_id}-{aug_type}/`, plus a
`summary.txt` with per-stage tensor statistics (shape, min/max/mean, unique values).

`synth_tag` is derived from `approach`: `native`, `normal`, or `synthFCD`. On the `normal`
approach, `synth-*` files are suppressed — they would be identical to the `real-*` files.

This is the fastest way to diagnose anything that looks wrong during training, and it is
what made most of the bugs listed in §9 findable at all.

---

## 7. Failure-Mode Analysis

### Per-Class Test Results

Pure-synthetic model, Bonn standard test split, n = 28:

| Class | Mean Dice |
| :--- | :---: |
| Background | 0.978 |
| White Matter | 0.834 |
| Cerebral Cortex | 0.772 |
| Deep Gray Matter | 0.686 |
| CSF | 0.669 |
| WM–GM Separator | 0.396 |
| **FCD Lesion** | **0.091** |

The healthy tissue classes confirm that domain-randomized synthesis transfers to real
FLAIR. The lesion class does not follow, and its standard deviation exceeds its mean — on
most test subjects the model produces nothing usable, and a small number of successes lift
the average off zero.

### `Dice_noFP` Oracle — False Positives Dominate

FCD Dice was recomputed after discarding every predicted lesion voxel outside the
ground-truth region, isolating the false-positive contribution.

| Metric | Value |
| :--- | :---: |
| FCD Dice as measured | 0.319 |
| FCD Dice with false positives removed | 0.649 |
| Relative improvement | **+104%** |

> **Provenance caveat.** These figures come from a validation-time experiment on an earlier
> checkpoint, not from the test-split evaluation above. The *relative* finding is the
> durable result; the absolute values should not be compared against the 0.091 test figure.

The competing hypothesis — that low-confidence FCD predictions were being suppressed by
argmax competition against the six other classes — is ruled out by this. Recall is not the
bottleneck; precision is.

Worth stating explicitly, because it is counterintuitive: in a 7-class softmax, FCD does
**not** need P(FCD) > 0.5 to win the argmax. It only needs to beat the other six classes,
which can happen at much lower probabilities. Argmax is not suppressing the lesion.

### Connected-Component Analysis — Negative Result

If the false positives were compact spurious blobs, keeping only the largest connected
component should recover most of the oracle gain.

| Post-processing | Δ FCD Dice |
| :--- | :---: |
| Largest connected component | **+0.003** |

Effectively null. The false positives are diffuse and spatially scattered, not
blob-separable, so no component-based filter removes them without also removing the lesion.

### Weighted Argmax — Negative Result

The FCD logit was scaled by a factor *W* before argmax, sweeping *W* from 1.0 to 5.0.
Dice declines **monotonically from W = 1.0 onward**.

Expected given the oracle result: upweighting the lesion class can only add false positives
when false positives are already the limiting factor. Experiments 1 and 3 form a consistent
picture and jointly close the argmax-suppression hypothesis.

### Leading Hypothesis

The residual gap looks like a **synthesis-realism problem**, not a training problem. The
FCD augmentations produce smooth, additive, locally-supported intensity changes; real FCD
lesions are textural and structurally irregular. A network trained on the former plausibly
learns "smooth local intensity anomaly" as its lesion prior, which fires broadly on real
FLAIR — exactly the diffuse false-positive pattern observed.

The network and training procedure behave correctly on the other six classes, which is what
localizes the problem to the generative model.

---

## 8. Things to Be Careful With

### Configuration

- **`flair_modality` defaults to `False`.** You must pass `--model.flair_modality true`
  explicitly for FLAIR training. There is no warning when it is missing.
- **Set `approach` on the model only.** `link_arguments` propagates it to data
  automatically. Never pass `--data.approach`, and never pass `native_synthesis` — it is a
  derived internal flag.
- **`fusedmask.nii` is required for `nativeSynth`.** `_scan()` silently drops any subject
  missing it. If zero subjects load, check that the dataset path includes the `/fcd/`
  component and that `fusedmask.nii` exists in every `sub-*` folder.
- **`lesion_gmm_params` only applies to `flair_modality=True` + `nativeSynth`.** On the
  standard path label 21 is pre-remapped to 19 and draws from the shared GMM; the parameter
  has no effect.
- **`normal` + `use_extra_data=True` is wrong.** Generated subjects have no real lesion
  annotation. A warning prints — do not ignore it.

### Checkpoints

- **Never put `save_last=True` on a metric-monitored `ModelCheckpoint`.** A monitored
  callback only writes when its metric improves, so `last.ckpt` freezes at the metric's
  last-best epoch. This cost 120 epochs of training before it was noticed. Use a dedicated
  unmonitored callback with `save_top_k=0`.
- **`save_hyperparameters()` bakes hparams into checkpoints.** Adding or renaming an
  `__init__` parameter breaks Lightning resume. Patch the `hyper_parameters` dict (see §5)
  or use `save_hyperparameters(ignore=[...])`.
- **`--ckpt_path` requires optimizer state.** Use it only for true resume. A weights-only
  warm start resets the epoch counter to 0 and drops optimizer state — never mix the two in
  one launch.
- **Recovering from a frozen `last.ckpt`:** point `--ckpt_path` at the most recent
  `best-dice-fcd-epoch=NNN-*.ckpt`, which is a full checkpoint.

### Synthesis

- **`nearest_if_label` requires integer dtype.** Cornucopia activates nearest-neighbour
  interpolation only when `not x.dtype.is_floating_point`. Pass `v.round().long()`, never
  `v.round().long().float()`. Getting this wrong applies cubic interpolation to categorical
  labels — it collapsed the WM–GM separator class from 167k to 3k voxels before it was
  caught.
- **Always call `make_final` before applying a deformation field.** It freezes the field to
  the input's spatial shape. Without it, output shapes vary across subjects and crash the
  UNet decoder.
- **`QuantileTransform` needs `clip=True`.** The default leaves the lower tail negative,
  and negatives entering `x^gamma` produce NaN. The `make_final(x)(x)` pattern is required —
  it is a `NonFinalTransform` and cannot be called directly.
- **`slab` and `rlab` will differ subtly at boundaries.** `slab` comes from the argmax of a
  deformed one-hot (smooth boundaries); `rlab` from nearest-neighbour deformation of the raw
  integer label map (hard boundaries). Expected and correct.
- **`FCDAugmentations` takes pre-sampled scalars.** Passing `sigma_range=`, `zoom_range=`,
  or `intensity_range=` is the old API and will crash.
- **Do not scale up `fcd_intensity_range`.** The values from `FCDParameterCalculator`
  (≈ 0.02–0.36) are relative contrast ratios used as absolute additive values. They are
  deliberately subtle because real FCD hyperintensity is low-contrast. Inflating them
  produces lesions the network learns trivially and then fails to find in real data.
- **`FLAIR_CLASS_PARAMS` is a module-level mutable global.** `_set_subject_params` shallow-
  copies before injecting class 21, so the global is never mutated. Safe because synthesis
  runs on the main process under `autocast(enabled=False)`.

### Python and Tooling

- **`[:None]` is a no-op slice.** When `n_tracked_batches` resolved to `None` on resume from
  an older checkpoint, `_val_batch_cache` and `_val_worst_cache` grew without bound.
- **`#` in a backslash-joined shell command comments out everything after it**, including
  all subsequent `\`-joined lines. A `# {ckpt_arg} \` line collapsed an entire launch
  command down to `python train... fit` with no arguments, and the script silently ran on
  defaults. Use an empty string for conditionally-absent flags:
  ```python
  ckpt_arg = f'--ckpt_path "{CKPT_PATH}"' if CKPT_PATH else ""
  ```
  Symptoms of this bug: "No seed found, seed set to 0" despite `--seed_everything 0`; the
  FCD parameter scan running despite ranges being supplied; OOM at a batch size you did not
  set.
- **Kaggle clones fresh from GitHub every run.** Local edits are invisible until pushed.
  Verify branch state before debugging a behaviour discrepancy:
  ```bash
  git clone <repo> /tmp/_check && grep -n "<pattern>" /tmp/_check/<file>
  ```
  Stale branches caused silent mismatches repeatedly.

### Evaluation

- **`clean_val=True` breaks historical comparability.** All earlier Dice numbers are
  augmented-real. Clean numbers run higher and smoother. Treat the switch as a new baseline.
- **A binary head's threshold does not transfer from 7-class.** Seeding a binary
  (background vs FCD) head from 7-class channel 6 transfers feature weights but not the
  decision threshold — logsumexp competition is not sigmoid-at-zero. Epoch-0 FCD Dice lands
  around 0.01 regardless of pretrained quality. Prefer lesion-weighted 7-class fine-tuning
  with `FocalTverskyLoss`.

---

## 9. Change History

Consolidated from the development session log. Ordered most recent first.

### Loss Function — Focal-Tversky

`FocalTverskyLoss` added as an alternative to `DiceCELoss`, applied to existing runs via
checkpoint patching (§5) that removes stale hyperparameters and rewrites the stored `loss`
key. Motivation: an asymmetric β penalizing false positives targets the dominant failure
mode identified in §7 more directly than symmetric Dice.

### `approach: str` Refactor

`native_synthesis: bool` replaced by `approach: str = 'synthFCD'` across `FCDDataset`,
`FCDDataModule`, `SharedSynth`, and `Model`, adding the `normal` control approach.
`link_arguments` updated to propagate `model.approach → data.approach`.

### `clean_val` Added

New `Model` parameter with `_clean_real_pair()` helper. Default `False` preserves existing
behaviour. See §4.11.

### `last.ckpt` Freeze Fixed

`save_last=True` was attached to the monitored `eval_loss` callback, freezing `last.ckpt`
at epoch 540 while training continued to 660. Fixed by moving `save_last` to a dedicated
unmonitored callback.

### `losses.py` — Two Confirmed `CatLoss` Bugs

Both identified as causes of prior training collapses. At batch size 2, a single bad subject
carries 50% of the gradient, making loss-function instability disproportionately dangerous.

**Bug 1 — `log(0)` loophole.** `pred.log()` on a zero probability produced `-inf`, which
`masked_fill_(~torch.isfinite(pred), 0)` silently converted to `0`, giving the network zero
penalty for a completely wrong prediction. Fixed with `pred.clamp(min=1e-7, max=1.0)` before
`log()`.

**Bug 2 — incorrect negation.** `forward_labels` ended with `loss = 1 - loss` instead of
`loss.neg_()`. Not standard CE negation, and numerically incorrect for the label-map path.

Also added: `ref_sum.clamp_min(1e-5)` on denominators in both `forward_labels` and
`forward_onehot`, and a `.float()` cast on `ref1`.

### LR Scheduler Fixed

`ReduceLROnPlateau` was stepping on `eval_loss` with `mode='min'` instead of `val_dice` with
`mode='max'`, so it never fired — for 500+ epochs. Compounded by Lightning ignoring the
scheduler dict entirely under `automatic_optimization=False`, requiring a manual
`sch.step(...)` call in `on_validation_epoch_end`. After the fix, the scheduler reduced LR
to the `min_lr=1e-6` floor within roughly 200 epochs.

### Validation Diagnostics Overhaul

Per-subject metrics, worst-batch tracking alongside best-batch, wide-format
`subject_metrics.csv`, subdirectory NIfTI layout, configurable
`val_diagnostics_interval`, and `n_best_batches` renamed to `n_tracked_batches`.

### Normalization Fixes (v52 Review)

A teammate's v52 introduced a slowdown traced to `IntensityTransform` running twice per
sample. The `rimg` call was correct in intent but paired with a wrong divisor (218.0). Fixed
by normalizing `rimg` via `QuantileTransform(clip=True)`, and by moving `simg` normalization
out of `_process_single_sample` and into `SynthFromLabelTransform` where the raw GMM output
belongs.

### Deformation Fixes

- Deformation was entirely skipped when `no_augs=True` — the gate covered both deformation
  and intensity. Removed; `no_augs` now disables intensity only.
- `frozen_deform = self.deform.make_final(x)` introduced for shape consistency.
- Cubic and nearest-neighbour deformation split across intensity and label volumes using the
  same frozen field.
- `remap_labels` was truncating instead of rounding: `lut[label_map.long()]` turned an
  interpolated 17.6 into 17 (background) instead of 18 (WM–GM class 5). Fixed to
  `lut[label_map.float().round().long()]`.
- `slab_out` was computed from the original undeformed `slab` and was never deformed. Fixed
  by capturing the deformed one-hot output and taking its argmax.
- `rimg` was returned as 3D `(D, H, W)` with no channel dimension, so `torch.stack`
  interpreted depth as channels and crashed the UNet with shape `(1, 256, 320, 16, 16)`.

### `native_synthesis` Implementation

Full end-to-end implementation of the native SynthSeg path: 22-channel one-hot, label 21
GMM injection, 21→19 pre-remap on the standard path, `roi` made optional in `SharedSynth`,
and `fusedmask.nii` added to the required-files scan.

### `modules.py` — `ConvBlockBase` Refactor

- Fixed `'stride' in opt_conv.get('stride', 1) != 1` — checking whether a string is `in` an
  integer raises `TypeError` at runtime for any separable strided convolution.
- Fixed caller-dict mutation via `opt_conv.pop(...)` by shallow-copying first.
- `InstanceNorm` given `eps=1e-4` for stability at small batch sizes.
- `make_norm` handles group norm and raises clear errors instead of silently returning
  `None`.

### Reverted Features

Built and then deliberately removed. Documented so they are not rediscovered by accident.

- **`lesion_only: bool`** — a binary specialization stage reshaping the head from 7 to 2
  channels. Reverted because epoch-0 FCD Dice landed at ~0.01 rather than ~0.2: the binary
  head must relearn its decision threshold from scratch (see §8).
- **`init_weights: Optional[str]`** — weights-only warm start bypassing Lightning's
  `--ckpt_path`. Reverted alongside `lesion_only`.
- **DDP on dual T4** — explored, rolled back.

FCD index is back to a hardcoded `6` / `[:, 6]` throughout.

---

## 10. Open Issues

| Issue | Status |
| :--- | :--- |
| **FCD Dice is not competitive** — 0.091 pure-synthetic, ≈ 0.14 fine-tuned, vs 0.256 supervised | Primary open problem. See §7 for the leading hypothesis. |
| **Lesion texture realism** — smooth additive augmentations may be the root cause of diffuse false positives | Untested. The highest-value next experiment. |
| **Transmantle ventricle label indices** — the transformation depends on indices (4 / 43) that may not match the SuperSynth scheme | Flagged, unverified. |
| **Label/appearance mismatch in thickening and transmantle** — both modify image content without a corresponding label update | Known, unresolved. |
| **Focal-Tversky vs DiceCE** — whether the asymmetric loss measurably reduces false positives | Evaluation incomplete. |
| **`max_epochs` ignored** — `--trainer.max_epochs` may not reach the trainer when `instantiate_trainer` overrides kwargs | Needs a diagnostic print to confirm. |
| **File ordering in `train_non_parametric_synthFCD.py`** | Cosmetic. |

### Suggested Next Experiments

1. **Texture-based lesion synthesis.** Replace the smoothed additive intensity boost with
   patch transplantation from real lesions, or with a learned texture model. Evaluate
   against `Dice_noFP` rather than raw Dice — the diagnostic is more sensitive to the
   mechanism being targeted.
2. **Asymmetric Focal-Tversky sweep.** Vary β to penalize false positives directly, and
   report precision and recall separately rather than Dice alone.
3. **Synthetic pretraining for the supervised pipeline.** The most promising practical use
   of this work may not be pure-synthetic training but large-scale pretraining for
   [nnU-FCD](https://github.com/YassienTawfikk/nnU-FCD).
4. **Resolve the label/appearance mismatch** before running further synthesis experiments —
   any conclusion drawn from thickening or transmantle subjects is currently confounded by
   it.
