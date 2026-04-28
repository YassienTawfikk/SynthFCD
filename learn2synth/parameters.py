import os
import re
import glob
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom as scipy_zoom


class FCDParameterCalculator:
    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def get_subj_num(self, subject_path):
        match = re.search(r'(\d+)', os.path.basename(subject_path))
        return int(match.group(1)) if match else 0

    def resample_to_target(self, source_arr, target_shape, is_labels=False):
        zoom_factors = [target_shape[i] / source_arr.shape[i]
                        for i in range(len(source_arr.shape))]
        return scipy_zoom(source_arr, zoom_factors, order=0 if is_labels else 1)

    # ------------------------------------------------------------------
    # Per-subject measurements
    # ------------------------------------------------------------------
    def calc_intensity_diff(self, flair_arr, roi_mask, label_arr):
        """
        Returns the relative FLAIR contrast of the FCD lesion vs surrounding tissue.
        Positive → hyperintense (expected for FCD on FLAIR). Returns None if unmeasurable.
        """
        if roi_mask.sum() == 0:
            return None

        # Find the label tissue type that overlaps most with the ROI
        dominant_label = None
        best_overlap = 0
        for tissue_id in np.unique(label_arr[roi_mask]):
            if tissue_id == 0:
                continue
            overlap = np.sum(roi_mask & (label_arr == tissue_id))
            if overlap > best_overlap:
                best_overlap, dominant_label = overlap, tissue_id

        if dominant_label is None:
            return None

        tissue_mask = label_arr == dominant_label
        lesion_mask = roi_mask & tissue_mask
        surround_mask = tissue_mask & ~roi_mask

        if lesion_mask.sum() == 0 or surround_mask.sum() == 0:
            return None

        mean_lesion = np.mean(flair_arr[lesion_mask])
        mean_surround = np.mean(flair_arr[surround_mask])

        # Scale-independent relative contrast
        rel_contrast = (mean_lesion - mean_surround) / (abs(mean_surround) + 1e-6)
        return float(rel_contrast)

    def calc_tail_len(self, roi_mask, label_arr, cortex_labels, ventricle_labels, wm_labels):
        """
        Estimates the transmantle tail length (voxels) from the WM lesion tip
        to the nearest ventricle surface, constrained to the ROI.
        Returns None if any required structure is absent.
        """
        ventricle_mask = np.isin(label_arr, ventricle_labels)
        cortex_mask = np.isin(label_arr, cortex_labels)
        wm_mask = np.isin(label_arr, wm_labels)

        required = [roi_mask, ventricle_mask, cortex_mask, wm_mask]
        if any(m.sum() == 0 for m in required):
            return None
        if (roi_mask & cortex_mask).sum() == 0 or (roi_mask & wm_mask).sum() == 0:
            return None

        ventricle_coords = np.argwhere(ventricle_mask)
        cortex_coords = np.argwhere(cortex_mask)
        roi_wm_coords = np.argwhere(roi_mask & wm_mask)
        roi_coords = np.argwhere(roi_mask)

        # WM point in ROI closest to cortex → lesion tip
        dists_to_cortex = [np.min(np.linalg.norm(cortex_coords - pt, axis=1)) for pt in roi_wm_coords]
        lesion_tip = roi_wm_coords[np.argmin(dists_to_cortex)]

        # ROI point closest to ventricle → tail root
        dists_to_vent = [np.min(np.linalg.norm(ventricle_coords - pt, axis=1)) for pt in roi_coords]
        tail_root = roi_coords[np.argmin(dists_to_vent)]

        return float(np.linalg.norm(lesion_tip - tail_root))

    # ------------------------------------------------------------------
    # Dataset-level parameter estimation
    # ------------------------------------------------------------------
    def calculate_fcd_parameters(self,
                                 dataset_path,
                                 label_file,
                                 flair_file,
                                 roi_file,
                                 intensity_subjects=None,
                                 transmantle_subjects=None,
                                 auto_resample=True,
                                 cortex_labels=[3, 42],
                                 ventricle_labels=[4, 43],
                                 wm_labels=[2, 41]):
        """
        Scans the dataset and returns data-driven (intensity_range, tail_length_range)
        tuples. Falls back to clinical defaults if no valid subjects are found.
        """
        intensity_diffs = []
        tail_lengths = []

        subject_folders = glob.glob(os.path.join(dataset_path, "*"))
        print(f"[FCD Params] Scanning {len(subject_folders)} folders…")

        for subject_folder in subject_folders:
            label_path = os.path.join(subject_folder, label_file)
            flair_path = os.path.join(subject_folder, flair_file)
            roi_path = os.path.join(subject_folder, roi_file)

            if not all(os.path.exists(p) for p in [label_path, flair_path, roi_path]):
                continue

            subject_num = self.get_subj_num(subject_folder)
            use_intensity = intensity_subjects is not None and subject_num in intensity_subjects
            use_trans = transmantle_subjects is not None and subject_num in transmantle_subjects

            if not use_intensity and not use_trans:
                continue

            try:
                flair_arr = nib.load(flair_path).get_fdata()
                label_arr = nib.load(label_path).get_fdata().astype(int)
                roi_arr = nib.load(roi_path).get_fdata().astype(int)

                # Resample label and ROI to FLAIR space if needed
                if flair_arr.shape != roi_arr.shape:
                    if not auto_resample:
                        continue
                    roi_arr = (self.resample_to_target(roi_arr, flair_arr.shape, True) > 0.5).astype(int)
                    label_arr = self.resample_to_target(label_arr, flair_arr.shape, True).astype(int)

                roi_mask = roi_arr > 0

                if use_intensity:
                    diff = self.calc_intensity_diff(flair_arr, roi_mask, label_arr)
                    if diff is not None:
                        intensity_diffs.append(diff)

                if use_trans:
                    tail = self.calc_tail_len(roi_mask, label_arr,
                                              cortex_labels, ventricle_labels, wm_labels)
                    if tail is not None:
                        tail_lengths.append(tail)

            except Exception as e:
                print(f"  [FCD Params] Skipping {os.path.basename(subject_folder)}: {e}")

        intensity_range = self._compute_intensity_range(intensity_diffs)
        tail_length_range = self._compute_tail_range(tail_lengths)

        print(f"[FCD Params] intensity_range    = {intensity_range}")
        print(f"[FCD Params] tail_length_range  = {tail_length_range}")
        return intensity_range, tail_length_range

    # ------------------------------------------------------------------
    # Range computation helpers
    # ------------------------------------------------------------------
    def _compute_intensity_range(self, intensity_diffs):
        """
        Derives a clipped percentile range from relative FLAIR contrasts.
        Only hyperintense (positive) observations are used.
        """
        if not intensity_diffs:
            print("[FCD Params] No intensity diffs found — using defaults.")
            return (0.05, 0.35)

        arr = np.array(intensity_diffs)
        arr_pos = arr[arr > 0]

        if len(arr_pos) >= 2:
            p5, p95 = float(np.percentile(arr_pos, 5)), float(np.percentile(arr_pos, 95))
        elif len(arr_pos) == 1:
            p5, p95 = float(arr_pos[0]) * 0.5, float(arr_pos[0]) * 1.5
        else:
            return (0.05, 0.35)  # all hypointense — clinical defaults

        lo = max(0.02, min(p5, 0.50))
        hi = max(lo + 0.05, min(p95, 0.70))
        return (round(lo, 4), round(hi, 4))

    def _compute_tail_range(self, tail_lengths):
        """
        Derives a clipped percentile range from transmantle tail length measurements.
        Guarantees a minimum spread of 10 voxels.
        """
        if not tail_lengths:
            print("[FCD Params] No tail lengths found — using defaults.")
            return (20, 50)

        arr = np.array(tail_lengths)

        if len(arr) >= 2:
            p5, p95 = float(np.percentile(arr, 5)), float(np.percentile(arr, 95))
        else:
            p5, p95 = float(arr[0]) * 0.8, float(arr[0]) * 1.2

        tmin = max(5, int(round(p5)))
        tmax = min(200, int(round(p95)))

        if tmax - tmin < 10:  # guarantee a meaningful spread
            tmin = max(5, tmin - 5)
            tmax = tmax + 5

        return (tmin, tmax)
