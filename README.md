# SynthFCD: Synthetic Data Generation for FCD II Lesion Segmentation in FLAIR MRI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: OpenNeuro](https://img.shields.io/badge/Dataset-OpenNeuro%20ds004199-blue.svg)](https://openneuro.org/datasets/ds004199)
[![Status: Research in Progress](https://img.shields.io/badge/Status-Research%20in%20Progress-orange.svg)](#results)

<p align="center">
  <img src="https://github.com/user-attachments/assets/0f1af2cc-a6bf-49d8-994b-8570a9cb9bbc" alt="SynthFCD Pipeline" width="100%">
  <br>
</p>

## Overview

The only public FCD Type II benchmark contains 85 patients. Every supervised method on this dataset — including our own [nnU-FCD](https://github.com/YassienTawfikk/nnU-FCD) — is limited by data scarcity before it is limited by anything else. SynthFCD tests whether that bottleneck can be removed by generating training images instead of collecting them.

> **This is an active research repository.** SynthFCD does **not** currently outperform the supervised nnU-Net baseline. It is released as a complete, reproducible pipeline together with an honest account of where it falls short.

---


**PyTorch implementation** of SynthFCD — a SynthSeg-derived pipeline that trains a 3D UNet to segment Focal Cortical Dysplasia lesions using FLAIR volumes synthesized on-the-fly from anatomical label maps, without using real images in the training branch.

<p align="center">
  <img src="https://github.com/user-attachments/assets/9400bf14-f919-4131-b103-8f29618e7159" alt="SynthFCD Pipeline" width="100%">
  <br>
  <em>Figure: The SynthFCD pipeline. A label map is deformed, converted to an image by per-class Gaussian Mixture sampling, given FCD lesion appearance, and corrupted with bias field, gamma, and noise before reaching the network.</em>
</p>

---

## What We're Looking For

FCD Type II lesions are subtle and heterogeneous. Four radiological features characterize them on FLAIR, and a lesion may show any combination of them.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8b43d088-5d2c-4a36-8f9b-1db351d44d98" alt="Radiological features of FCD Type II" width="100%">
  <br>
  <em>Figure: Cortical thickening, hyperintensity, transmantle sign, and blurring of the gray–white matter boundary. Yellow boxes mark the lesion.</em>
</p>

---

## What We Generate

Each training step builds a fresh image from a label map: tissue labels are sampled into intensities by a per-subject Gaussian Mixture Model, then the FCD lesion is created — either by injecting appearance into the lesion region, or by giving the lesion its own intensity distribution as a first-class label.

Lesion synthesis is conditioned on radiology: each subject is assigned the feature its real lesion actually exhibits, so the synthetic lesion distribution mirrors the real one.

<p align="center">
  <img src="https://github.com/user-attachments/assets/1549c05d-5437-4949-8ead-cb76dd390555" alt="Real versus synthetic FCD features" width="100%">
  <br>
  <em>Figure: Real MRI (top) versus synthesized FLAIR (bottom) for each of the four radiological features.</em>
</p>

---

## Results

Evaluated on the Bonn FCD II standard test split (n = 28).

<!-- <p align="center">
  <img src="PLACEHOLDER_PER_CLASS_DICE" alt="Per-class Dice on the test split" width="100%">
  <br>
  <em>Figure: Per-class mean Dice ± std for the pure-synthetic model. Healthy tissue classes transfer well; the lesion class does not.</em>
</p> -->

The result splits cleanly in two. A network that has never seen a real training image segments real FLAIR anatomy competently — white matter at 0.83, cortex at 0.77, deep gray matter and CSF near 0.69. Domain-randomized synthesis transfers.

The lesion class does not follow. FCD Dice reaches 0.091, with a standard deviation larger than the mean: on most test subjects the model finds nothing usable, and the few successes are what lift the average off zero.

| Approach | Training data | FCD Dice (test) |
| :--- | :--- | :---: |
| SynthFCD (pure synthetic) | synthetic only | 0.091 |
| SynthFCD + real fine-tuning | synthetic → real | ≈ 0.14 |
| [nnU-FCD](https://github.com/YassienTawfikk/nnU-FCD) (supervised) | real only | **0.256** |

Fine-tuning on real images improves the lesion class but does not close the gap to supervised training.

The error is concentrated in false positives rather than missed lesions — an oracle that discards every prediction outside the true lesion region roughly doubles FCD Dice, so the model is locating the lesion and then also flagging a great deal of healthy tissue. Connected-component filtering and weighted argmax were both tested as remedies and both failed, which points at lesion realism in the generative model rather than at the network or the training procedure.

<p align="center">
  <img src="https://github.com/user-attachments/assets/0b208a53-5c76-4aa1-a374-8bdf207a9991" alt="Qualitative results" width="100%">
  <br>
  <em>Figure: Real FLAIR, ground truth, prediction, and P(FCD) probability map.</em>
</p>

---

## Getting Started

```bash
git clone https://github.com/YassienTawfikk/SynthFCD.git
cd SynthFCD
pip install -r requirements.txt
```

Download the [Bonn FCD II dataset](https://openneuro.org/datasets/ds004199) and run `notebooks/00_Dataset_Preparation.ipynb` to build the per-subject layout.

```bash
python train_non_parametric_synthFCD.py fit \
  --model.approach synthFCD \
  --model.flair_modality true \
  --data.dataset_path data/fcd/ \
  --trainer.max_epochs 2000 \
  --trainer.accelerator gpu
```

<details>
<summary><b>Other training modes and options</b></summary>

<br>

| Mode | Flag | Description |
| :--- | :--- | :--- |
| SynthFCD | `--model.approach synthFCD` | Lesion appearance injected after synthesis |
| Native SynthSeg | `--model.approach nativeSynth` | Lesion as a first-class GMM label |
| Normal | `--model.approach normal` | Direct supervision on real FLAIR (control) |

```bash
# resume training
--ckpt_path experiments/<run>/checkpoints/last.ckpt

# dump every synthesis stage as NIfTI for named subjects
--model.debug_subject_ids '["sub-00001", "sub-00033"]'
```

Requires `cornucopia` pinned to `6f8ab58`, and PyTorch ≤ 2.4 on P100 hardware.

</details>

---

## Future Work

- **Lesion texture realism** — replacing smooth additive perturbations with texture synthesis, the change most likely to reduce false positives.
- **Precision-targeted losses** — asymmetric Focal-Tversky evaluated against the false-positive diagnostic rather than raw Dice.
- **Synthetic pretraining** — using this pipeline to initialize the supervised nnU-Net rather than to replace it.

---

## Documentation

Full technical record — architecture, design decisions, known issues, and pitfalls — in [`docs/SynthFCD_Documentation.md`](docs/SynthFCD_Documentation.md).

Built on [SynthSeg](https://github.com/BBillot/SynthSeg) (Billot et al.) and [cornucopia](https://github.com/balbasty/cornucopia) (Balbastre).

---

## Citation

```bibtex
@misc{Tawfik2026SynthFCD,
  title  = {SynthFCD: Synthetic Data Generation for FCD II Lesion Segmentation in FLAIR MRI},
  author = {Tawfik, Yassien and Marwan, Mazen and Yasser, Mohamed and
            Mahmoud, Nancy and Mosaad, Madonna and Salman, Mahmoud and
            Basha, Tamer and Khalaf, Aya},
  year   = {2026},
  note   = {Department of Systems and Biomedical Engineering, Cairo University},
  howpublished = {\url{https://github.com/YassienTawfikk/SynthFCD}}
}
```

## Authors

<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://medicine.yale.edu/profile/aya-khalaf/" target="_blank">
          <img src="https://ysm-res.cloudinary.com/image/upload/c_crop,x_354,y_0,w_2396,h_2400/c_fill,f_auto,q_auto:eco,dpr_2,w_650/v1/yms/prod/4426ecb7-0a4d-4e6d-af35-c329e1ae6e54" width="190px;" alt="Aya Khalaf"/>
          <br/><br/>
          <sub><b>Dr. Aya Khalaf</b></sub>
        </a>
        <br/>
        <sub>Yale University</sub>
      </td>
    </tr>
  </table>
</div>
<br/>
<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://github.com/YassienTawfikk" target="_blank">
          <img src="https://avatars.githubusercontent.com/u/126521373?v=4" width="140px;" alt="Yassien Tawfik"/>
          <br/>
          <sub><b>Yassien Tawfik</b></sub>
        </a>
        <br/>
        <sub>Cairo University</sub>
      </td>
      <td align="center">
        <a href="https://github.com/mohamedddyasserr" target="_blank">
          <img src="https://avatars.githubusercontent.com/u/126451832?v=4" width="140px;" alt="Mohamed Yasser"/>
          <br/>
          <sub><b>Mohamed Yasser</b></sub>
        </a>
        <br/>
        <sub>Cairo University</sub>
      </td>
      <td align="center">
        <a href="https://github.com/nancymahmoud1" target="_blank">
          <img src="https://avatars.githubusercontent.com/u/125357872?v=4" width="140px;" alt="Nancy Mahmoud"/>
          <br/>
          <sub><b>Nancy Mahmoud</b></sub>
        </a>
        <br/>
        <sub>Cairo University</sub>
      </td>
      <td align="center">
        <a href="https://github.com/Mazenmarwan023" target="_blank">
          <img src="https://avatars.githubusercontent.com/u/127551364?v=4" width="140px;" alt="Mazen Marwan"/>
          <br/>
          <sub><b>Mazen Marwan</b></sub>
        </a>
        <br/>
        <sub>Cairo University</sub>
      </td>      
      <td align="center">
        <a href="https://github.com/madonna-mosaad" target="_blank">
          <img src="https://avatars.githubusercontent.com/u/127048836?v=4" width="140px;" alt="Madonna Mosaad"/>
          <br/>
          <sub><b>Madonna Mosaad</b></sub>
        </a>
        <br/>
        <sub>Cairo University</sub>
      </td>
    </tr>
  </table>
</div>
<br/>
<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://github.com/mahmoud1yaser" target="_blank">
          <img src="https://avatars.githubusercontent.com/u/88372358?v=4" width="165px;" alt="Mahmoud Salman"/>
          <br/>
          <sub><b>Mahmoud Salman</b></sub>
        </a>
        <br/>
        <sub>Western University</sub>
      </td>
    </tr>
  </table>
</div>
<br/>
<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://www.linkedin.com/in/tamer-basha-b81812ab/" target="_blank">
          <img src="https://media.licdn.com/dms/image/v2/C5103AQEkkCY9JaaHTQ/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1517586051833?e=1772668800&v=beta&t=yluL2Xa1N5UEb7w8EEiatadwA9xM7KPOzutf05yJkMI" width="135px;" alt="Tamer Basha"/>
          <br/>
          <sub><b>Tamer Basha</b></sub>
        </a>
        <br/>
        <sub>Cairo University</sub>
      </td>
    </tr>
  </table>
</div>
