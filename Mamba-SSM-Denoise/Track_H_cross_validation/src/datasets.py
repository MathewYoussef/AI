import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import pandas as pd


@dataclass
class SpectraDatasetConfig:
    """Container for dataset-related hyperparameters."""

    sequence_length: int
    train_dir: str
    clean_dir: Optional[str] = None
    noise2noise: bool = False
    pairwise_noise2noise: bool = False
    target_wavelengths: Optional[Sequence[float]] = None
    wavelength_key: str = "wavelength_nm"
    reflectance_key: str = "reflectance"
    dtype: np.dtype = np.float32
    split: str = "train"
    manifest_path: Optional[str] = None
    dose_features_path: Optional[str] = None
    film_features: Sequence[str] = ("cos_theta",)
    cond_mean: Optional[Sequence[float]] = None
    cond_std: Optional[Sequence[float]] = None
    sampling_weights_path: Optional[str] = None


class SpectraDataset(Dataset):
    """Loads UV reflectance spectra with geometric context features.

    Each sample returns a 3-channel input:
      1. Reflectance in [0, 1]
      2. Normalised wavelength λ_norm ∈ [-1, 1]
      3. Angle feature (scalar repeated across the sequence)

    Targets are either supervised clean spectra or pseudo-clean Noise2Noise
    targets obtained by averaging the remaining replicates from the same
    (treatment, sample, angle) group.
    """

    def __init__(
        self,
        config: SpectraDatasetConfig,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.cfg = config
        self.transform = transform
        self.target_transform = target_transform
        self.root_dir = Path(self.cfg.train_dir).resolve()
        self.split = self.cfg.split
        self.pairwise_noise2noise = self.cfg.pairwise_noise2noise and self.cfg.noise2noise
        self.rng = np.random.default_rng()

        self.film_features = tuple(self.cfg.film_features)
        self.cond_mean = (
            np.asarray(self.cfg.cond_mean, dtype=np.float32)
            if self.cfg.cond_mean is not None
            else None
        )
        self.cond_std = (
            np.asarray(self.cfg.cond_std, dtype=np.float32)
            if self.cfg.cond_std is not None
            else None
        )

        self.lambda_grid = self._initialise_wavelength_grid()
        self.lambda_norm = torch.from_numpy(self._normalise_wavelengths(self.lambda_grid))
        self.lambda_step = float(self.lambda_grid[1] - self.lambda_grid[0]) if self.lambda_grid.size > 1 else 1.0

        self.metadata_rows: List[Dict[str, str]] = []
        self.relative_paths: List[str] = []
        self.noisy_files = self._load_file_index(self.root_dir)
        if not self.noisy_files:
            raise ValueError(f"No spectra found in {self.cfg.train_dir}")

        self.clean_files = None
        if self.cfg.clean_dir is not None:
            self.clean_files = self._build_clean_map(self.cfg.clean_dir, self.noisy_files)

        if self.cfg.noise2noise and self.clean_files is not None:
            raise ValueError("Noise2Noise and clean supervision are mutually exclusive")
        if self.cfg.noise2noise and len(self.noisy_files) < 2:
            raise ValueError("Noise2Noise mode requires at least two spectra")

        self.cache: List[np.ndarray] = [None] * len(self.noisy_files)
        self.group_sum: Optional[Dict[str, np.ndarray]] = {} if (self.cfg.noise2noise and not self.pairwise_noise2noise) else None
        self.group_count: Optional[Dict[str, int]] = {} if (self.cfg.noise2noise and not self.pairwise_noise2noise) else None
        self.group_angle: Dict[str, float] = {}
        self.group_to_indices: Dict[str, List[int]] = {}
        self.index_to_group: List[str] = []
        self.index_to_sample: List[str] = []
        self.sample_to_indices: Dict[str, List[int]] = {}
        self.index_to_replicate: List[int] = []

        self._pair_stats_enabled = False
        self._pair_stats_total_pairs: int = 0
        self._pair_stats_violations: int = 0
        self._pair_stats_distance: Dict[int, int] = defaultdict(int)
        self._pair_stats_partner_usage: Dict[str, set[int]] = defaultdict(set)

        self._build_group_index()
        self.conditioning: List[torch.Tensor] = []
        self.sample_weights: Optional[torch.Tensor] = None
        self._initialise_conditioning()

    def _initialise_wavelength_grid(self) -> np.ndarray:
        if self.cfg.target_wavelengths is not None:
            grid = np.asarray(self.cfg.target_wavelengths, dtype=self.cfg.dtype)
        else:
            grid_path = self.root_dir / "wavelength_grid.npy"
            if grid_path.exists():
                grid = np.load(grid_path).astype(self.cfg.dtype)
            else:
                start = 300.0
                stop = 600.0
                if self.cfg.sequence_length > 1:
                    grid = np.linspace(start, stop, self.cfg.sequence_length, dtype=self.cfg.dtype)
                else:
                    grid = np.array([start], dtype=self.cfg.dtype)
            self.cfg.target_wavelengths = grid.tolist()
        return grid

    @staticmethod
    def _normalise_wavelengths(lambda_grid: np.ndarray) -> np.ndarray:
        if lambda_grid.size <= 1:
            return np.zeros_like(lambda_grid, dtype=np.float32)
        min_val = float(lambda_grid[0])
        max_val = float(lambda_grid[-1])
        return ((2.0 * (lambda_grid - min_val) / (max_val - min_val)) - 1.0).astype(np.float32)

    def _load_file_index(self, root: Path) -> List[str]:
        manifest_path = (
            Path(self.cfg.manifest_path).resolve()
            if self.cfg.manifest_path is not None
            else root / "manifest.csv"
        )
        files: List[str] = []
        if manifest_path.exists():
            manifest_df = pd.read_csv(manifest_path)
            for _, row in manifest_df.iterrows():
                rel_path = str(row["relative_path"]).strip()
                abs_path = (root / rel_path).resolve()
                if not abs_path.exists():
                    raise FileNotFoundError(f"Manifest path {abs_path} does not exist")
                files.append(str(abs_path))
                self.relative_paths.append(rel_path)
                row_dict = row.to_dict()
                self.metadata_rows.append(
                    {
                        "relative_path": rel_path,
                        "treatment": str(row_dict.get("treatment", "")),
                        "sample": str(row_dict.get("sample", "")),
                        "angle": str(row_dict.get("angle", "")),
                        "group_id": str(row_dict.get("group_id", "")),
                    }
                )
        else:
            patterns = ("*.npy", "*.npz", "*.csv", "*.txt")
            for pattern in patterns:
                for file_path in sorted(root.rglob(pattern)):
                    if file_path.name in {"wavelength_grid.npy", "manifest.csv", "lambda_stats.npz"}:
                        continue
                    files.append(str(file_path))
            for path_str in files:
                abs_path = Path(path_str).resolve()
                try:
                    rel_path = str(abs_path.relative_to(root))
                except ValueError:
                    rel_path = abs_path.name
                self.relative_paths.append(rel_path)
                parts = Path(rel_path).parts
                treatment = parts[0] if parts else ""
                sample = parts[1] if len(parts) > 1 else ""
                angle = parts[2] if len(parts) > 2 else ""
                self.metadata_rows.append(
                    {
                        "relative_path": rel_path,
                        "treatment": treatment,
                        "sample": sample,
                        "angle": angle,
                        "group_id": "",
                    }
                )
        return files

    @staticmethod
    def angle_to_cos(angle_name: str) -> float:
        match = re.match(r"(\d+)", str(angle_name))
        if not match:
            return 0.0
        hour = int(match.group(1)) % 12
        radians = 2.0 * math.pi * hour / 12.0
        return float(math.cos(radians))

    @staticmethod
    def _build_clean_map(clean_dir: str, noisy_files: Sequence[str]) -> List[str]:
        clean_map: List[str] = []
        for noisy_path in noisy_files:
            filename = os.path.basename(noisy_path)
            candidate = os.path.join(clean_dir, filename)
            if not os.path.exists(candidate):
                raise FileNotFoundError(
                    f"Missing clean target for {filename}; expected {candidate}"
                )
            clean_map.append(candidate)
        return clean_map

    def _build_group_index(self) -> None:
        for idx, path_str in enumerate(self.noisy_files):
            path = Path(path_str).resolve()
            try:
                relative_parent = path.parent.relative_to(self.root_dir)
            except ValueError:
                relative_parent = path.parent
            group = str(relative_parent)

            spectrum = self._load_spectrum(path_str)
            self.cache[idx] = spectrum

            self.group_to_indices.setdefault(group, []).append(idx)
            self.index_to_group.append(group)

            sample_root = str(Path(group).parent)
            self.index_to_sample.append(sample_root)
            self.sample_to_indices.setdefault(sample_root, []).append(idx)

            replicate_match = re.search(r"rep_(\d+)", path.name)
            replicate_id = int(replicate_match.group(1)) if replicate_match else idx
            self.index_to_replicate.append(replicate_id)

            if self.cfg.noise2noise and not self.pairwise_noise2noise:
                if group not in self.group_sum:
                    self.group_sum[group] = spectrum.copy()
                    self.group_count[group] = 1
                else:
                    self.group_sum[group] += spectrum
                    self.group_count[group] += 1

            if group not in self.group_angle:
                angle_value = self.angle_to_cos(Path(group).name)
                self.group_angle[group] = angle_value

        if self.cfg.noise2noise:
            if self.pairwise_noise2noise:
                too_small = [group for group, indices in self.group_to_indices.items() if len(indices) < 2]
            else:
                too_small = [group for group, count in self.group_count.items() if count < 2]
            if too_small:
                sample_groups = ", ".join(sorted(too_small[:5]))
                raise ValueError(
                    "Noise2Noise grouping requires at least two captures per group; "
                    f"groups with insufficient samples: {sample_groups}"
                )

    def _initialise_conditioning(self) -> None:
        dose_lookup: Dict[str, Dict[str, float]] = {}
        if self.cfg.dose_features_path is not None:
            dose_df = pd.read_csv(self.cfg.dose_features_path)
            if "split" in dose_df.columns:
                dose_df = dose_df[dose_df["split"] == self.split]
            dose_lookup = {
                str(row["relative_path"]): row.to_dict()
                for _, row in dose_df.iterrows()
            }

        if self.cfg.sampling_weights_path is not None:
            weights_df = pd.read_csv(self.cfg.sampling_weights_path)
            weight_map = {
                str(row["relative_path"]): float(row["sampling_weight"])
                for _, row in weights_df.iterrows()
            }
            weights = [weight_map.get(rel, 1.0) for rel in self.relative_paths]
            self.sample_weights = torch.tensor(weights, dtype=torch.double)

        self.conditioning.clear()
        for idx, rel_path in enumerate(self.relative_paths):
            meta = self.metadata_rows[idx]
            angle_label = meta.get("angle") or Path(self.index_to_group[idx]).name
            cos_theta = float(self.angle_to_cos(angle_label))

            cond_components: List[float] = []
            dose_row = dose_lookup.get(rel_path)

            for feature in self.film_features:
                key = feature.strip().lower()
                if key == "cos_theta":
                    cond_components.append(cos_theta)
                elif key in {"uva_total", "u_total"}:
                    value = float(dose_row.get("UVA_total_mWh_cm2", 0.0)) if dose_row else 0.0
                    cond_components.append(value)
                elif key in {"uvb_total", "v_total"}:
                    value = float(dose_row.get("UVB_total_mWh_cm2", 0.0)) if dose_row else 0.0
                    cond_components.append(value)
                elif key in {"uva_hours", "uva_hours_h", "exposure_uva_hours"}:
                    value = float(dose_row.get("UVA_hours_h", 0.0)) if dose_row else 0.0
                    cond_components.append(value)
                elif key in {"uvb_hours", "uvb_hours_h", "exposure_uvb_hours"}:
                    value = float(dose_row.get("UVB_hours_h", 0.0)) if dose_row else 0.0
                    cond_components.append(value)
                elif key in {"p_uva", "p_uva_mw_cm2", "uva_power"}:
                    value = float(dose_row.get("P_UVA_mW_cm2", 0.0)) if dose_row else 0.0
                    cond_components.append(value)
                elif key in {"p_uvb", "p_uvb_mw_cm2", "uvb_power"}:
                    value = float(dose_row.get("P_UVB_mW_cm2", 0.0)) if dose_row else 0.0
                    cond_components.append(value)
                elif key == "uva_over_uvb":
                    value = float(dose_row.get("UVA_over_UVB", 0.0)) if dose_row else 0.0
                    if not np.isfinite(value):
                        value = 0.0
                    cond_components.append(value)
                elif key == "uva_norm":
                    value = float(dose_row.get("UVA_norm", 0.0)) if dose_row else 0.0
                    cond_components.append(value)
                elif key == "uvb_norm":
                    value = float(dose_row.get("UVB_norm", 0.0)) if dose_row else 0.0
                    cond_components.append(value)
                elif key == "uva_total_z":
                    value = float(dose_row.get("UVA_total_z", 0.0)) if dose_row else 0.0
                    cond_components.append(value)
                elif key == "uvb_total_z":
                    value = float(dose_row.get("UVB_total_z", 0.0)) if dose_row else 0.0
                    cond_components.append(value)
                else:
                    raise KeyError(f"Unsupported conditioning feature '{feature}'")

            if not cond_components:
                cond_components.append(cos_theta)

            cond_raw = np.asarray(cond_components, dtype=np.float32)
            if self.cond_mean is not None and self.cond_std is not None:
                if self.cond_mean.shape[0] != cond_raw.shape[0] or self.cond_std.shape[0] != cond_raw.shape[0]:
                    raise ValueError(
                        "Conditioning statistics shape mismatch with requested features"
                    )
                cond_norm = (cond_raw - self.cond_mean) / (self.cond_std + 1e-6)
            else:
                cond_norm = cond_raw
            cond_norm = np.nan_to_num(cond_norm, nan=0.0, posinf=0.0, neginf=0.0)
            self.conditioning.append(torch.from_numpy(cond_norm.astype(np.float32)))

    def _resample(self, wavelength: np.ndarray, spectrum: np.ndarray) -> np.ndarray:
        target = np.asarray(self.cfg.target_wavelengths, dtype=self.cfg.dtype)
        if wavelength.shape[0] == target.shape[0] and np.allclose(wavelength, target):
            return spectrum.astype(self.cfg.dtype, copy=False)
        resampled = np.interp(target, wavelength, spectrum)
        return resampled.astype(self.cfg.dtype, copy=False)

    def _load_csv(self, path: str) -> np.ndarray:
        data = np.genfromtxt(path, delimiter=",", names=True)
        if self.cfg.target_wavelengths is not None:
            wl = np.asarray(data[self.cfg.wavelength_key], dtype=self.cfg.dtype)
            refl = np.asarray(data[self.cfg.reflectance_key], dtype=self.cfg.dtype)
            spectrum = self._resample(wl, refl)
        else:
            spectrum = np.asarray(data[self.cfg.reflectance_key], dtype=self.cfg.dtype)
        return np.clip(spectrum, 0.0, 1.0)

    def _load_npy(self, path: str) -> np.ndarray:
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            if "reflectance" in data:
                spectrum = np.asarray(data["reflectance"], dtype=self.cfg.dtype)
            else:
                raise KeyError(f"Expected 'reflectance' key in {path}")
        else:
            spectrum = np.asarray(data, dtype=self.cfg.dtype)
        return np.clip(spectrum, 0.0, 1.0)

    def _load_spectrum(self, path: str) -> np.ndarray:
        if path.endswith(".csv") or path.endswith(".txt"):
            spectrum = self._load_csv(path)
        else:
            spectrum = self._load_npy(path)

        if spectrum.ndim != 1:
            raise ValueError(f"Spectrum at {path} must be 1-D; got shape {spectrum.shape}")

        if spectrum.shape[0] != self.cfg.sequence_length:
            raise ValueError(
                f"Expected length {self.cfg.sequence_length} but got {spectrum.shape[0]}"
            )
        return spectrum.astype(self.cfg.dtype, copy=False)

    def __len__(self) -> int:
        return len(self.noisy_files)

    def _noise2noise_target(self, index: int, group: str) -> np.ndarray:
        if self.group_sum is None or self.group_count is None:
            raise RuntimeError("Noise2Noise statistics not initialised")
        count = self.group_count[group]
        if count <= 1:
            raise ValueError(f"Group {group} has insufficient replicates ({count})")
        spectrum = self.cache[index]
        target = (self.group_sum[group] - spectrum) / (count - 1)
        return target.astype(self.cfg.dtype, copy=False)

    def _pairwise_noise2noise_target(self, index: int, group: str) -> np.ndarray:
        candidates = self.group_to_indices.get(group, [])
        if len(candidates) < 2:
            raise ValueError(f"Group {group} has insufficient replicates ({len(candidates)})")
        partner_candidates = [idx for idx in candidates if idx != index]
        partner_idx = int(self.rng.choice(partner_candidates))
        partner = self.cache[partner_idx]
        if partner is None:
            partner = self._load_spectrum(self.noisy_files[partner_idx])
            self.cache[partner_idx] = partner
        self._record_pair_stats(index, partner_idx, group)
        return partner.astype(self.cfg.dtype, copy=False)

    def enable_pair_stats(self, enabled: bool = True) -> None:
        self._pair_stats_enabled = bool(enabled)
        if enabled:
            self.reset_pair_stats()
        else:
            self.reset_pair_stats()

    def reset_pair_stats(self) -> None:
        self._pair_stats_total_pairs = 0
        self._pair_stats_violations = 0
        self._pair_stats_distance = defaultdict(int)
        self._pair_stats_partner_usage = defaultdict(set)

    def _record_pair_stats(self, anchor_idx: int, partner_idx: int, group: str) -> None:
        if not self._pair_stats_enabled:
            return

        self._pair_stats_total_pairs += 1

        anchor_sample = self.index_to_sample[anchor_idx]
        partner_sample = self.index_to_sample[partner_idx]
        anchor_angle = Path(self.index_to_group[anchor_idx]).name
        partner_angle = Path(self.index_to_group[partner_idx]).name

        if partner_sample != anchor_sample or partner_angle != anchor_angle:
            self._pair_stats_violations += 1
            return

        anchor_rep = self.index_to_replicate[anchor_idx]
        partner_rep = self.index_to_replicate[partner_idx]
        distance = abs(partner_rep - anchor_rep)
        self._pair_stats_distance[distance] += 1

        usage_key = partner_rep
        self._pair_stats_partner_usage[group].add(usage_key)

    def pair_stats_summary(self) -> Dict[str, object]:
        if not self._pair_stats_enabled:
            return {}

        coverage_per_group: Dict[str, float] = {}
        coverage_values: List[float] = []
        for group, indices in self.group_to_indices.items():
            total = len(indices)
            used = self._pair_stats_partner_usage.get(group, set())
            coverage = (len(used) / total) if total else 0.0
            coverage_pct = 100.0 * coverage
            coverage_per_group[group] = coverage_pct
            coverage_values.append(coverage_pct)

        coverage_median = float(np.median(coverage_values)) if coverage_values else None

        distance_hist = {str(distance): count for distance, count in self._pair_stats_distance.items()}

        return {
            "total_pairs": self._pair_stats_total_pairs,
            "violations": self._pair_stats_violations,
            "distance_hist": distance_hist,
            "coverage_per_group_pct": coverage_per_group,
            "coverage_values_pct": coverage_values,
            "coverage_median_pct": coverage_median,
        }

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spectrum = self.cache[index]
        if spectrum is None:
            spectrum = self._load_spectrum(self.noisy_files[index])
            self.cache[index] = spectrum

        reflectance = torch.from_numpy(spectrum.astype(np.float32))
        lambda_norm = self.lambda_norm

        group = self.index_to_group[index]
        angle_value = self.group_angle.get(group, 0.0)
        angle_channel = torch.full_like(reflectance, fill_value=angle_value)

        features = torch.stack((reflectance, lambda_norm, angle_channel), dim=0)

        if self.clean_files is not None:
            target_array = self._load_spectrum(self.clean_files[index])
        elif self.cfg.noise2noise:
            if self.pairwise_noise2noise:
                target_array = self._pairwise_noise2noise_target(index, group)
            else:
                target_array = self._noise2noise_target(index, group)
        else:
            target_array = spectrum

        target = torch.from_numpy(target_array.astype(np.float32)).unsqueeze(0)

        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            target = self.target_transform(target)

        cond = self.conditioning[index]

        return features, target, cond


def create_dataloader(
    cfg: SpectraDatasetConfig,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    drop_last: bool = False,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    balance_groups: bool = False,
    prefetch_factor: int = 2,
) -> DataLoader:
    dataset = SpectraDataset(cfg)
    sampler = None
    if dataset.sample_weights is not None:
        sampler = WeightedRandomSampler(
            dataset.sample_weights,
            num_samples=len(dataset.sample_weights),
            replacement=True,
        )
    elif balance_groups:
        if len(dataset.sample_to_indices) == 0:
            raise ValueError("Unable to balance groups because no samples were discovered")
        weights = torch.zeros(len(dataset), dtype=torch.double)
        for indices in dataset.sample_to_indices.values():
            weight = 1.0 / max(len(indices), 1)
            weights[indices] = weight
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers and num_workers > 0,
    )
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    return DataLoader(dataset, **loader_kwargs)
