import torch
import torch.nn.functional as F
import math
import random


class FCDAugmentations:
    def gaussian_blur_3d_torch(self, tensor_3d: torch.Tensor, sigma: float,
                               device: torch.device) -> torch.Tensor:
        """
        Separable 3-D Gaussian blur implemented entirely in PyTorch.
        Runs on *device* (GPU or CPU). Returns a (Z,Y,X) float tensor.

        Parameters
        ----------
        tensor_3d : torch.Tensor  shape (Z, Y, X)
        sigma     : float – Gaussian standard deviation in voxels
        device    : torch.device
        """
        if sigma < 1e-4:
            return tensor_3d.clone()

        # Build a 1-D unnormalised Gaussian kernel
        radius = int(math.ceil(3 * sigma))
        ks = 2 * radius + 1  # odd kernel size
        coords = torch.arange(ks, dtype=torch.float32, device=device) - radius
        kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        t = tensor_3d.float().unsqueeze(0).unsqueeze(0)  # (1,1,Z,Y,X)

        # Convolve along Z
        kz = kernel_1d.view(1, 1, ks, 1, 1)
        t = F.conv3d(t, kz, padding=(radius, 0, 0))
        # Convolve along Y
        ky = kernel_1d.view(1, 1, 1, ks, 1)
        t = F.conv3d(t, ky, padding=(0, radius, 0))
        # Convolve along X
        kx = kernel_1d.view(1, 1, 1, 1, ks)
        t = F.conv3d(t, kx, padding=(0, 0, radius))

        return t.squeeze(0).squeeze(0)

    def binary_dilation_torch(self, mask_3d: torch.Tensor,
                              iterations: int = 3,
                              device: torch.device = None) -> torch.Tensor:
        """
        Morphological dilation of a binary (Z,Y,X) mask using F.max_pool3d.
        Equivalent to scipy.ndimage.binary_dilation(mask, iterations=iterations).
        Runs on *device* (GPU or CPU). Returns a float32 (Z,Y,X) tensor.
        """
        t = mask_3d.float().unsqueeze(0).unsqueeze(0)  # (1,1,Z,Y,X)
        for _ in range(iterations):
            t = F.max_pool3d(t, kernel_size=3, stride=1, padding=1)
        return t.squeeze(0).squeeze(0)

    def apply_roi_augmentations_blured(self, synthetic, roi, seed=None,
                                       sigma_range=(0.3, 1.0)):
        """
        Apply random Gaussian blur inside ROI mask – fully GPU-native.

        Parameters
        ----------
        synthetic : torch.Tensor  (Z, Y, X)  – lives on GPU
        roi       : torch.Tensor  (Z, Y, X)  – binary / integer mask
        seed      : int, optional
        sigma_range : tuple(float, float)

        Returns
        -------
        augmented : torch.Tensor  (Z, Y, X)  – stays on GPU
        """
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        device = synthetic.device
        augmented = synthetic.clone()
        roi_mask = (roi > 0).float().to(device)

        sigma = random.uniform(*sigma_range)
        blurred = self.gaussian_blur_3d_torch(augmented, sigma, device)

        # Blend blurred region back only inside the ROI
        augmented = augmented * (1.0 - roi_mask) + blurred * roi_mask
        return augmented

    def apply_roi_thickening(self, synthetic, roi, seed=None, zoom_range=(0.2, 0.4),
                             bound='border'):
        """
        Simulate cortical thickening by zooming only the ROI region with random zoom factor.

        Parameters
        ----------
        synthetic : torch.Tensor (Z,Y,X)
            Input synthetic image.
        roi : torch.Tensor (Z,Y,X)
            ROI mask.
        seed : int, optional
            Random seed for reproducibility.
        zoom_range : tuple(float,float)
            Range of zoom factors to sample from.
            >1 expands ROI (thickening), <1 shrinks ROI (thinning).
        bound : str
            Padding mode for grid_sample.

        Returns
        -------
        out : torch.Tensor
            Synthetic image with randomized ROI thickening.
        """
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        device = synthetic.device
        dtype = torch.float32

        img = synthetic.to(dtype).unsqueeze(0).unsqueeze(0)  # (1,1,Z,Y,X)
        mask = (roi > 0).to(dtype).unsqueeze(0).unsqueeze(0)

        Z, Y, X = synthetic.shape

        # Compute ROI center
        nz = torch.nonzero(mask.squeeze())
        if nz.numel() == 0:
            return synthetic
        cz, cy, cx = nz.float().mean(dim=0).tolist()

        # --- Random zoom factor ---
        zoom_factor = random.uniform(*zoom_range)

        # Build affine zoom matrix about ROI center
        E = torch.eye(4, device=device, dtype=dtype)
        T1 = E.clone();
        T1[:3, 3] = torch.tensor([-cz, -cy, -cx], device=device)
        S = E.clone();
        S[0, 0] = zoom_factor;
        S[1, 1] = zoom_factor;
        S[2, 2] = zoom_factor
        T2 = E.clone();
        T2[:3, 3] = torch.tensor([cz, cy, cx], device=device)
        A = T2 @ S @ T1

        # Build grid
        z = torch.linspace(0, Z - 1, Z, device=device, dtype=dtype)
        y = torch.linspace(0, Y - 1, Y, device=device, dtype=dtype)
        x = torch.linspace(0, X - 1, X, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        ones = torch.ones_like(xx)
        coords = torch.stack([zz, yy, xx, ones], dim=-1)
        mapped = coords @ A.T
        mz, my, mx = mapped[..., 0], mapped[..., 1], mapped[..., 2]
        gx = 2 * (mx / (X - 1)) - 1;
        gy = 2 * (my / (Y - 1)) - 1;
        gz = 2 * (mz / (Z - 1)) - 1
        grid = torch.stack([gx, gy, gz], dim=-1)[None, ...]

        # Warp image
        warped_img = F.grid_sample(img, grid, mode='bilinear',
                                   padding_mode=bound, align_corners=True)
        # Warp mask with the SAME grid (nearest-neighbor to keep it binary)
        warped_mask = F.grid_sample(mask, grid, mode='nearest',
                                    padding_mode='zeros', align_corners=True)

        # Blend: use warped_mask so the boundary follows the zoomed tissue
        out = img * (1 - warped_mask) + warped_img * warped_mask

        return out.squeeze(0).squeeze(0), warped_mask.squeeze(0).squeeze(0)

    def apply_roi_augmentations_hyperintensity(self,
                                               synthetic, roi, seed=None,
                                               intensity_range=(0.1, 0.5), sigma_range=(0.0, 0.6)
                                               ):
        """
        Apply random hyperintensity augmentation to ROI region – fully GPU-native.

        Parameters
        ----------
        synthetic       : torch.Tensor  (Z, Y, X)  – lives on GPU
        roi             : torch.Tensor  (Z, Y, X)  – binary / integer mask
        seed            : int, optional
        intensity_range : tuple(float, float)
        sigma_range     : tuple(float, float)

        Returns
        -------
        augmented : torch.Tensor  (Z, Y, X)  – stays on GPU
        """
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        device = synthetic.device
        augmented = synthetic.clone()

        # ROI binary mask – stays on device
        roi_mask_t = (roi > 0).float().to(device)

        intensity_factor = random.uniform(*intensity_range)
        sigma = random.uniform(*sigma_range)

        # Smooth the ROI mask to make the hyperintensity look realistic
        hyper_map = self.gaussian_blur_3d_torch(roi_mask_t, sigma, device)

        # Add the hyperintensity map (all on GPU, no .cpu() / .numpy())
        augmented = augmented + hyper_map.to(augmented.dtype) * intensity_factor
        return augmented

    def apply_roi_augmentations_transmantle(self,
                                            synthetic, roi, labeled_image, seed=None,
                                            tail_length_range=(20, 50),
                                            intensity_range=(0.1, 0.5),
                                            sigma_range=(0.0, 0.6),
                                            tail_dilation_iterations=1
                                            ):
        """
        Simulate transmantle sign – fully GPU-native.
        Hyperintense tail from cortex ROI toward nearest lateral ventricle (labels 4 / 43).

        Parameters
        ----------
        synthetic               : torch.Tensor  (Z, Y, X)  – lives on GPU
        roi                     : torch.Tensor  (Z, Y, X)  – ROI mask
        labeled_image           : torch.Tensor  (Z, Y, X)  – atlas labels
        seed                    : int, optional
        tail_length_range       : tuple(int, int)
        intensity_range         : tuple(float, float)
        sigma_range             : tuple(float, float)
        tail_dilation_iterations: int  – controls tail thickness (default 1, was 3)

        Returns
        -------
        augmented          : torch.Tensor  (Z, Y, X)  – stays on GPU
        roi_mask_with_tail : torch.Tensor  (Z, Y, X, float32)  – stays on GPU
        """
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        device = synthetic.device
        augmented = synthetic.clone()

        # ── ROI centroid (on GPU) ────────────────────────────────────────────────
        roi_mask_t = (roi > 0).float().to(device)  # (Z,Y,X)
        roi_coords = torch.nonzero(roi_mask_t, as_tuple=False).float()  # (N,3)
        if roi_coords.numel() == 0:
            return augmented, roi_mask_t
        roi_centroid = roi_coords.mean(dim=0)  # (3,)

        # ── Ventricle centroids (on GPU) ─────────────────────────────────────────
        lab = labeled_image.to(device)
        vent_left_coords = torch.nonzero(lab == 4, as_tuple=False).float()  # (M,3)
        vent_right_coords = torch.nonzero(lab == 43, as_tuple=False).float()  # (K,3)

        c_left = vent_left_coords.mean(dim=0) if vent_left_coords.numel() > 0 else None
        c_right = vent_right_coords.mean(dim=0) if vent_right_coords.numel() > 0 else None

        chosen_centroid = None
        if c_left is not None and c_right is not None:
            d_left = torch.linalg.norm(roi_centroid - c_left)
            d_right = torch.linalg.norm(roi_centroid - c_right)
            chosen_centroid = c_left if d_left < d_right else c_right
        elif c_left is not None:
            chosen_centroid = c_left
        elif c_right is not None:
            chosen_centroid = c_right

        # ── Build tail mask toward ventricle (on GPU) ─────────────────────────────
        Z, Y, X = synthetic.shape
        tail_mask = torch.zeros(Z, Y, X, dtype=torch.float32, device=device)

        if chosen_centroid is not None:
            direction = chosen_centroid - roi_centroid  # (3,)
            noise = torch.randn(3, device=device) * 0.2
            direction = direction + noise
            direction = direction / (torch.linalg.norm(direction) + 1e-6)

            steps = random.randint(*tail_length_range)
            step_range = torch.arange(1, steps, dtype=torch.float32, device=device)  # (steps-1,)
            offsets_f = roi_centroid.unsqueeze(0) + direction.unsqueeze(0) * step_range.unsqueeze(1)  # (steps-1, 3)
            offsets = offsets_f.long()  # (steps-1, 3)

            # Keep only coordinates inside the volume
            valid = (
                    (offsets[:, 0] >= 0) & (offsets[:, 0] < Z) &
                    (offsets[:, 1] >= 0) & (offsets[:, 1] < Y) &
                    (offsets[:, 2] >= 0) & (offsets[:, 2] < X)
            )
            offsets = offsets[valid]  # (M, 3)
            if offsets.numel() > 0:
                tail_mask[offsets[:, 0], offsets[:, 1], offsets[:, 2]] = 1.0

            # Morphological dilation – thickness controlled by tail_dilation_iterations
            tail_mask = self.binary_dilation_torch(tail_mask, iterations=tail_dilation_iterations, device=device)

        # Union of ROI and tail
        combined_mask = torch.clamp(roi_mask_t + tail_mask, 0.0, 1.0)  # (Z,Y,X) float32

        # ── Random hyperintensity (on GPU) ────────────────────────────────────────
        intensity_factor = random.uniform(*intensity_range)
        sigma = random.uniform(*sigma_range)

        hyper_map = self.gaussian_blur_3d_torch(combined_mask, sigma, device)
        augmented = augmented + hyper_map.to(augmented.dtype) * intensity_factor

        return augmented, combined_mask
