import torch
import cornucopia as cc


def donothing(x):
    return x


# ---------------------------------------------------------------------------
# FLAIR-like per-class intensity parameters (remapped label space 0–17)
# ---------------------------------------------------------------------------
FLAIR_CLASS_PARAMS: dict = {
    0: {"mu": (0.0, 13.81), "sigma": (1.37, 19.6)},  # Background
    1: {"mu": (109.84, 223.42), "sigma": (13.84, 36.84)},  # White Matter
    2: {"mu": (116.44, 209.95), "sigma": (19.78, 47.53)},  # Cortex
    3: {"mu": (97.5, 225.03), "sigma": (13.94, 39.36)},  # Deep GM / PV
    4: {"mu": (39.66, 172.89), "sigma": (29.73, 67.62)},  # CSF
    5: {"mu": (86.69, 223.39), "sigma": (12.96, 53.23)},  # Optic chiasm
    6: {"mu": (16.81, 101.83), "sigma": (21.09, 84.7)},  # Air Internal
    7: {"mu": (20.27, 138.59), "sigma": (15.07, 79.41)},  # Artery
    8: {"mu": (15.14, 160.24), "sigma": (11.79, 58.54)},  # Eye balls
    9: {"mu": (50.31, 143.43), "sigma": (41.79, 82.53)},  # Other tissues
    10: {"mu": (87.97, 177.37), "sigma": (26.2, 67.31)},  # Rectus muscles
    11: {"mu": (66.42, 199.57), "sigma": (26.21, 74.48)},  # Mucosa
    12: {"mu": (62.25, 152.26), "sigma": (36.28, 79.03)},  # Skin
    13: {"mu": (16.81, 211.53), "sigma": (7.23, 60.49)},  # Spinal cord
    14: {"mu": (20.08, 146.72), "sigma": (25.18, 58.34)},  # Vein
    15: {"mu": (25.22, 133.92), "sigma": (30.2, 67.11)},  # Bone cortical
    16: {"mu": (61.11, 203.46), "sigma": (34.42, 77.68)},  # Bone cancellous
    17: {"mu": (71.88, 165.96), "sigma": (20.83, 62.5)},  # Optic nerve
}


# ---------------------------------------------------------------------------
# Load per-subject class params from CSV (produced by extract_flair_stats)
# ---------------------------------------------------------------------------

def load_subject_class_params(
        csv_path: str,
        subject_id: str,
        fallback: dict | None = None,
) -> dict:
    """Build a per-class GMM parameter dict for one subject from a CSV.

    Reads ``flair_stats_raw.csv`` (produced by the extract_flair_stats
    notebook) and returns a dict compatible with
    :class:`PerClassGaussianMixtureTransform`::

        {class_id: {"mu": (mu_lo, mu_hi), "sigma": (sigma_lo, sigma_hi)}, ...}

    Parameters
    ----------
    csv_path : str
        Path to ``flair_stats_raw.csv``.
    subject_id : str
        Subject folder name as it appears in the CSV ``subject`` column
        (e.g. ``"sub-00010"``).  Leading/trailing whitespace is ignored.
    fallback : dict, optional
        Dict used for classes absent in the subject's CSV rows.
        Defaults to the global ``FLAIR_CLASS_PARAMS``.

    Returns
    -------
    dict[int, dict]
        One entry per class (0-17) with ``mu`` and ``sigma`` tuples.
    """
    import pandas as pd

    if fallback is None:
        fallback = FLAIR_CLASS_PARAMS

    df = pd.read_csv(csv_path)

    # Normalise the subject column so name mismatches don't silently fail
    df["subject"] = df["subject"].astype(str).str.strip()
    subject_id = str(subject_id).strip()

    sub_df = df[df["subject"] == subject_id]
    if sub_df.empty:
        available = df["subject"].unique().tolist()
        raise ValueError(
            f"Subject '{subject_id}' not found in {csv_path}. Available subjects (first 10): {available[:10]}"
        )

    # Required columns (added by the updated extract_flair_stats notebook)
    required = {"class_id", "mu_lo", "mu_hi", "sigma_lo", "sigma_hi"}
    missing = required - set(sub_df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing columns: {missing}. Re-run extract_flair_stats_notebook with the updated version that computes per-subject bounds."
        )

    params: dict = {}
    for _, row in sub_df.iterrows():
        cls = int(row["class_id"])
        params[cls] = {
            "mu": (float(row["mu_lo"]), float(row["mu_hi"])),
            "sigma": (float(row["sigma_lo"]), float(row["sigma_hi"])),
        }

    # Fill in any classes the subject didn't have enough voxels for
    all_classes = set(range(18)) | set(fallback.keys())
    for cls in all_classes:
        if cls not in params:
            params[cls] = fallback.get(cls, {"mu": (0, 255), "sigma": (0, 16)})

    n_from_csv = len(sub_df)
    n_from_fallback = len(params) - n_from_csv
    print(
        f"[load_subject_class_params] '{subject_id}': {n_from_csv} classes from CSV, {n_from_fallback} filled from fallback."
    )
    return params


# ---------------------------------------------------------------------------
# Legacy global Gaussian mixture transform
# ---------------------------------------------------------------------------

class RandomGaussianMixtureTransform(torch.nn.Module):
    """Sample from a Gaussian mixture – one range for ALL tissue classes."""

    def __init__(self, mu=255, sigma=16, fwhm=2, background=None, dtype=None):
        super().__init__()
        self.dtype = dtype
        self.sample = dict(
            mu=cc.random.Uniform.make(cc.random.make_range(0, mu)),
            sigma=cc.random.Uniform.make(cc.random.make_range(0, sigma)),
            fwhm=cc.random.Uniform.make(cc.random.make_range(0, fwhm)),
        )
        self.background = background

    def forward(self, x):
        theta = self.get_parameters(x)
        return self.apply_transform(x, theta)

    def get_parameters(self, x):
        mu = self.sample["mu"](len(x))
        sigma = self.sample["sigma"](len(x))
        fwhm = int(self.sample["fwhm"]())
        if x.dtype.is_floating_point:
            backend = dict(dtype=x.dtype, device=x.device)
        else:
            backend = dict(dtype=self.dtype or torch.get_default_dtype(), device=x.device)
        mu = torch.as_tensor(mu).to(**backend)
        sigma = torch.as_tensor(sigma).to(**backend)
        return mu, sigma, fwhm

    def apply_transform(self, x, parameters):
        mu, sigma, fwhm = parameters
        backend = dict(dtype=mu.dtype, device=x.device)
        if self.background is not None:
            x[self.background] = 0
        y1 = torch.randn(*x.shape, **backend)
        y1 = cc.utils.conv.smoothnd(y1, fwhm=fwhm)
        y1 = y1.mul_(sigma[..., None, None, None]).add_(mu[..., None, None, None])
        y = torch.sum(x.to(**backend) * y1, dim=0)
        return y[None]


# ---------------------------------------------------------------------------
# Per-class Gaussian mixture transform
# ---------------------------------------------------------------------------

class PerClassGaussianMixtureTransform(torch.nn.Module):
    """Each tissue class gets its own (mu_min, mu_max) / (sigma_min, sigma_max).

    Parameters
    ----------
    class_params : dict[int, dict]
        Mapping from channel index to {"mu": (lo, hi), "sigma": (lo, hi)}.
    fwhm : float
        Upper bound for within-class smoothing FWHM.
    background : int, optional
        Channel index to zero out before synthesis.
    default_mu / default_sigma : tuple
        Fallback range for channels not listed in class_params.
    dtype : torch.dtype, optional
        Output dtype when input is integer.
    """

    def __init__(
            self,
            class_params: dict,
            fwhm: float = 2,
            background: int | None = None,
            default_mu: tuple = (0, 255),
            default_sigma: tuple = (0, 16),
            dtype=None,
    ):
        super().__init__()
        self.class_params = class_params
        self.default_mu = default_mu
        self.default_sigma = default_sigma
        self.background = background
        self.dtype = dtype
        self.fwhm_sampler = cc.random.Uniform.make(cc.random.make_range(0, fwhm))

    def _sample_scalar(self, lo: float, hi: float) -> float:
        if lo >= hi:
            return float(lo)
        return lo + (hi - lo) * torch.rand(1).item()

    def get_parameters(self, x):
        n_classes = len(x)
        mu_vals, sigma_vals = [], []
        for i in range(n_classes):
            p = self.class_params.get(i, None)
            mu_lo, mu_hi = p["mu"] if p else self.default_mu
            sigma_lo, sigma_hi = p["sigma"] if p else self.default_sigma
            mu_vals.append(self._sample_scalar(mu_lo, mu_hi))
            sigma_vals.append(self._sample_scalar(sigma_lo, sigma_hi))

        fwhm = int(self.fwhm_sampler())

        if x.dtype.is_floating_point:
            backend = dict(dtype=x.dtype, device=x.device)
        else:
            backend = dict(dtype=self.dtype or torch.get_default_dtype(), device=x.device)

        mu = torch.tensor(mu_vals, **backend)
        sigma = torch.tensor(sigma_vals, **backend)
        return mu, sigma, fwhm

    def apply_transform(self, x, parameters):
        mu, sigma, fwhm = parameters
        backend = dict(dtype=mu.dtype, device=x.device)

        if self.background is not None:
            x = x.clone()
            x[self.background] = 0

        y1 = torch.randn(*x.shape, **backend)
        if fwhm > 0:
            y1 = cc.utils.conv.smoothnd(y1, fwhm=fwhm)
        y1 = y1.mul_(sigma[..., None, None, None]).add_(mu[..., None, None, None])
        y = torch.sum(x.to(**backend) * y1, dim=0)
        return y[None]

    def forward(self, x):
        theta = self.get_parameters(x)
        return self.apply_transform(x, theta)


# ---------------------------------------------------------------------------
# SynthFromLabelTransform
# ---------------------------------------------------------------------------

class SynthFromLabelTransform(torch.nn.Module):
    """Synthesize an MRI from an existing one-hot label map.

    Parameters
    ----------
    class_params : dict, optional
        Per-class (mu, sigma) ranges for :class:`PerClassGaussianMixtureTransform`.
        Defaults to ``None`` → falls back to the legacy global GMM.
    use_per_class_gmm : bool
        When *True* and class_params is provided, use the per-class GMM.
    """

    def __init__(
            self,
            num_ch=1,
            patch=None,
            rotation=15,
            shears=0.012,
            zooms=0.15,
            elastic=0.05,
            elastic_nodes=10,
            gmm_fwhm=10,
            bias=7,
            gamma=0.6,
            motion_fwhm=3,
            resolution=8,
            snr=10,
            gfactor=5,
            order=3,
            skip_gmm=False,
            no_augs=False,
            class_params: dict | None = None,
            use_per_class_gmm: bool = True,
    ):
        super().__init__()
        self.no_augs = no_augs
        self.num_ch = num_ch

        # ---- Geometric deformation ----
        self.deform = donothing if no_augs else cc.RandomAffineElasticTransform(
            elastic,
            elastic_nodes,
            order=order,
            bound="zeros",
            rotations=rotation,
            shears=shears,
            zooms=zooms,
            patch=patch,
        )

        # ---- GMM ----
        if skip_gmm:
            self.gmm = None
        elif use_per_class_gmm and class_params is not None:
            self.gmm = PerClassGaussianMixtureTransform(
                class_params=class_params,
                fwhm=gmm_fwhm,
                background=0,
            )
        else:
            self.gmm = RandomGaussianMixtureTransform(fwhm=gmm_fwhm, background=0)

        # ---- Post-GMM intensity augmentations ----
        self.intensity = donothing if no_augs else cc.IntensityTransform(
            bias, gamma, motion_fwhm, resolution, snr, gfactor, order
        )

    # ------------------------------------------------------------------
    def forward(self, x, coreg=None):
        """
        Parameters
        ----------
        x : Tensor (n_classes, H, W, D)  – one-hot label map
        coreg : Tensor or list[Tensor], optional
            Extra volumes that must be co-deformed with x (e.g. FCD mask).

        Returns
        -------
        img  : Tensor (1, H, W, D)         – synthetic FLAIR
        lab  : Tensor (n_classes, H, W, D) – (deformed) label map
        coreg: Tensor or list[Tensor]       – only returned when coreg is not None
        """
        # 1. Geometric deformation
        #    Stack x + coreg so they all receive the SAME random field.
        if not self.no_augs:
            if coreg is not None:
                coreg_list = list(coreg) if isinstance(coreg, (list, tuple)) else [coreg]
                n_lab = x.shape[0]
                stacked = torch.cat([x] + coreg_list, dim=0)
                stacked = self.deform(stacked)
                x = stacked[:n_lab]
                deformed_co = [stacked[n_lab + i] for i in range(len(coreg_list))]
                coreg = deformed_co if isinstance(coreg, (list, tuple)) else deformed_co[0]
            else:
                x = self.deform(x)

        # 2. GMM synthesis
        if self.gmm is not None:
            gmm_params = [self.gmm.get_parameters(x) for _ in range(self.num_ch)]
            img = torch.cat(
                [self.intensity(self.gmm.apply_transform(x, gmm_params[i]))
                 for i in range(self.num_ch)],
                dim=0,
            )
        else:
            img = self.intensity(x)

        if coreg is not None:
            return img, x, coreg
        return img, x


# ---------------------------------------------------------------------------
# CCSynthSeg – dictionary-transform wrapper
# ---------------------------------------------------------------------------

class CCSynthSeg:
    """MONAI-style dict transform wrapping :class:`SynthFromLabelTransform`.

    Defaults to the FLAIR intensity preset (:data:`FLAIR_CLASS_PARAMS`).
    Pass ``use_per_class_gmm=False`` to revert to the legacy global GMM.
    """

    def __init__(
            self,
            label_key,
            image_key="image",
            coreg_keys=None,
            num_ch=1,
            patch=None,
            rotation=15,
            shears=0.012,
            zooms=0.15,
            elastic=0.05,
            elastic_nodes=10,
            gmm_fwhm=10,
            bias=7,
            gamma=0.6,
            motion_fwhm=3,
            resolution=8,
            snr=10,
            gfactor=5,
            order=3,
            skip_gmm=False,
            no_augs=False,
            class_params: dict | None = None,
            use_per_class_gmm: bool = True,
    ) -> None:
        self.label_key = label_key
        self.image_key = image_key
        self.coreg_keys = (
            coreg_keys if isinstance(coreg_keys, (tuple, list)) else [coreg_keys]
        )

        if class_params is None and use_per_class_gmm:
            class_params = FLAIR_CLASS_PARAMS

        self.transform = SynthFromLabelTransform(
            num_ch=num_ch,
            patch=patch,
            rotation=rotation,
            shears=shears,
            zooms=zooms,
            elastic=elastic,
            elastic_nodes=elastic_nodes,
            gmm_fwhm=gmm_fwhm,
            bias=bias,
            gamma=gamma,
            motion_fwhm=motion_fwhm,
            resolution=resolution,
            snr=snr,
            gfactor=gfactor,
            order=order,
            skip_gmm=skip_gmm,
            no_augs=no_augs,
            class_params=class_params,
            use_per_class_gmm=use_per_class_gmm,
        )

    def __call__(self, data):
        d = dict(data)
        result = self.transform(
            d[self.label_key],
            [d[key] for key in self.coreg_keys] if self.coreg_keys is not None else None,
        )
        if len(result) == 3:
            img, lab, coreg = result
        else:
            img, lab = result
            coreg = None

        d[self.image_key] = img
        d[self.label_key] = lab
        if self.label_key + "_meta_dict" in d:
            d[self.image_key + "_meta_dict"] = d[self.label_key + "_meta_dict"]
        if coreg is not None:
            for i, key in enumerate(self.coreg_keys):
                d[key] = coreg[i] if isinstance(coreg, list) else coreg
        d["mpm"] = torch.zeros_like(img)
        return d
