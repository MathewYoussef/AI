import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.models.mamba_uv import build_model


def load_wavelength_grid(path: Optional[str], sequence_length: int) -> np.ndarray:
    if path:
        grid = np.load(Path(path)).astype(np.float32)
        if grid.shape[0] != sequence_length:
            raise ValueError(
                f"Grid length {grid.shape[0]} != sequence length {sequence_length}"
            )
        return grid
    start = 300.0
    stop = 600.0
    if sequence_length > 1:
        return np.linspace(start, stop, sequence_length, dtype=np.float32)
    return np.array([start], dtype=np.float32)


def angle_to_scalar(angle_label: str) -> float:
    angle_label = angle_label.strip()
    digits = "".join(ch for ch in angle_label if ch.isdigit())
    if not digits:
        return 0.0
    hour = int(digits) % 12
    radians = 2.0 * math.pi * hour / 12.0
    return float(math.cos(radians))


def load_spectrum(path: str, sequence_length: int) -> torch.Tensor:
    if path.endswith(".csv"):
        data = np.genfromtxt(path, delimiter=",", names=True)
        if "reflectance" not in data.dtype.names:
            raise KeyError("CSV must contain a 'reflectance' column")
        spectrum = np.asarray(data["reflectance"], dtype=np.float32)
    else:
        spectrum = np.load(path).astype(np.float32)
    if spectrum.ndim != 1:
        raise ValueError(f"Expected 1-D spectrum, got shape {spectrum.shape}")
    if spectrum.shape[0] != sequence_length:
        raise ValueError(
            f"Sequence length mismatch: expected {sequence_length}, got {spectrum.shape[0]}"
        )
    spectrum = np.clip(spectrum, 0.0, 1.0)
    return torch.from_numpy(spectrum)


def build_input_tensor(
    reflectance: torch.Tensor,
    wavelength_grid: np.ndarray,
    angle_label: str,
) -> torch.Tensor:
    lambda_norm = torch.from_numpy(
        (2.0 * (wavelength_grid - wavelength_grid[0]) / (wavelength_grid[-1] - wavelength_grid[0]) - 1.0).astype(
            np.float32
        )
    )
    angle_scalar = angle_to_scalar(angle_label)
    angle_channel = torch.full_like(reflectance, fill_value=angle_scalar)
    return torch.stack((reflectance, lambda_norm, angle_channel), dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for Mamba-UV denoiser")
    parser.add_argument("--model", default="mamba_tiny_uv")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--infile", required=True, help="Input spectrum (.npy/.csv)")
    parser.add_argument("--outfile", default=None, help="Output .npy path")
    parser.add_argument("--sequence_length", "--L", type=int, default=601, dest="sequence_length")
    parser.add_argument("--wavelength_grid", default=None, help="Optional wavelength grid .npy")
    parser.add_argument("--angle", default="12Oclock", help="Angle label (e.g., 12Oclock or 6Oclock)")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(args.model, sequence_length=args.sequence_length, in_channels=3)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    wavelength_grid = load_wavelength_grid(args.wavelength_grid, args.sequence_length)
    spectrum = load_spectrum(args.infile, args.sequence_length)
    features = build_input_tensor(spectrum, wavelength_grid, args.angle).unsqueeze(0).to(device)

    with torch.no_grad():
        denoised = model(features).squeeze(0).squeeze(0)

    outfile = args.outfile or (
        Path(args.infile).with_suffix("")
        .with_name(Path(args.infile).stem + "_denoised.npy")
    )
    np.save(str(outfile), denoised.cpu().numpy())
    print(f"Saved denoised spectrum to {outfile}")


if __name__ == "__main__":
    main()
