# SynthFCD: Synthetic Data Generation for Focal Cortical Dysplasia Segmentation

**SynthFCD** is a deep learning framework designed to train robust segmentation models for Focal Cortical Dysplasia (FCD) using purely synthetic data. By leveraging a **Teacher-Student** architecture, this project generates realistic MRI scans with synthetic lesions on-the-fly, enabling the training of segmentation networks without the need for large, manually annotated datasets.

## 🚀 Key Features

*   **Teacher-Student Architecture:**
    *   **Teacher (Generator):** A synthesis module that takes label maps and applies random geometric deformations, intensity augmentations, and MRI artifacts (bias field, noise, motion) to create realistic synthetic images.
    *   **Student (Segmentor):** A U-Net based segmentation network that learns to predict the original labels from the synthetic images.
*   **Domain Randomization:** Extensive augmentation ensures the model generalizes well to real-world clinical data.
*   **FCD-Specific Simulation:** Custom deformation fields simulate the subtle structural abnormalities characteristic of FCD.
*   **Flexible Backbones:** Supports various segmentation architectures (U-Net, SegNet) and loss functions (Dice, Cross-Entropy).

## 🛠️ Project Structure

The repository is organized as follows:

```
SynthFCD/
├── scripts/                  # Training and utility scripts
│   ├── train_non_parametric_synthseg.py       # Original training script
│   ├── train_non_parametric_synthseg_V02.py   # Refactored training script (Recommended)
│   ├── train_non_parametric_synthseg_vanilla.py # Folder-based data loading
│   └── train_non_parametric_unet.py           # 9-class U-Net implementation
├── learn2synth/              # Core library (Neural Networks, Losses, Training Loops)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/SynthFCD.git
    cd SynthFCD
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

### Training

To start training the model using the recommended script:

```bash
python scripts/train_non_parametric_synthseg_V02.py
```

**Key Arguments:**
You can customize training via command-line arguments (handled by `pytorch_lightning.cli`):
*   `--data.images`: Path to image folder.
*   `--data.labels`: Path to label folder.
*   `--model.loss`: Loss function (default: `dice`).
*   `--trainer.max_epochs`: Number of epochs.

### Data Preparation
The scripts support two data loading conventions:
1.  **Subject-based** (used by `_V02.py`): Expects `sub-XXX/FusedMask.nii` and `sub-XXX/FLAIR.nii`.
2.  **Folder-based** (used by `_vanilla.py`): Expects `labels/*.nii` and matching `images/*.nii`.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## 📄 License

[Specify License Here, e.g., MIT, Apache 2.0]
