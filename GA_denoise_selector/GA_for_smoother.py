#!/usr/bin/env python3
"""Genetic algorithm optimiser for spectral denoising smoothers."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygad

from smoothers import (
    AVAILABLE_FAMILIES,
    BilateralParams,
    ButterworthParams,
    CascadeParams,
    GPParams,
    KalmanParams,
    MedianHampelParams,
    PCAParams,
    SGParams,
    WienerParams,
    SplineParams,
    TVParams,
    WaveletParams,
    apply_bilateral,
    apply_butterworth,
    apply_cascade,
    apply_gp,
    apply_kalman,
    apply_median_hampel,
    apply_pca,
    apply_sg,
    apply_spline,
    apply_tv,
    apply_wavelet,
    apply_whittaker,
    apply_wiener,
    bilateral_complexity,
    butterworth_complexity,
    cascade_complexity,
    gp_complexity,
    kalman_complexity,
    make_cascade_params,
    make_sg_params,
    make_wavelet_params,
    median_hampel_complexity,
    pca_complexity,
    sg_complexity,
    spline_complexity,
    tv_complexity,
    validate_bilateral_params,
    validate_butter_params,
    validate_gp_params,
    validate_kalman_params,
    validate_median_hampel_params,
    validate_pca_params,
    validate_sg_params,
    validate_spline_params,
    validate_tv_params,
    validate_wavelet_params,
    validate_whittaker_params,
    validate_wiener_params,
    wavelet_complexity,
    WhittakerParams,
    wiener_complexity,
    whittaker_complexity,
)


@dataclass
class SmootherSpec:
    name: str
    param_options: List[List]
    build: callable
    validate: callable
    apply: callable
    complexity: callable


@dataclass
class GAOutputs:
    best_genome: Sequence[float]
    fitness: float
    spectra_smoothed: np.ndarray
    metrics: Dict[str, float]
    smoother_name: str
    parameters: Dict[str, object]


# Parameter grids -----------------------------------------------------------
SG_ORDER_CHOICES = [2, 3, 4]
SG_WINDOW_CHOICES = [5, 7, 9, 11, 13]
WAVELET_LEVEL_CHOICES = [1, 2, 3, 4, 5, 6]
BUTTER_ORDER_CHOICES = [2, 3, 4, 5]
BUTTER_CUTOFF_CHOICES = [0.05, 0.1, 0.15, 0.2]
WIENER_WINDOW_CHOICES = [3, 5, 7, 9]
WIENER_NOISE_CHOICES = [1e-4, 5e-4, 1e-3]
MEDIAN_KERNEL_CHOICES = [3, 5, 7, 9, 11]
HAMPEL_WINDOW_CHOICES = [5, 7, 9]
HAMPEL_SIGMA_CHOICES = [3.0, 4.0, 5.0]
BILATERAL_COLOR_CHOICES = [0.02, 0.05, 0.1]
BILATERAL_SPATIAL_CHOICES = [1.0, 2.0, 4.0]
WHITTAKER_LAM_CHOICES = [1e2, 1e3, 1e4, 1e5]
WHITTAKER_D_CHOICES = [1, 2, 3]
SPLINE_S_CHOICES = [0.01, 0.05, 0.1, 0.2]
TV_WEIGHT_CHOICES = [0.01, 0.025, 0.05, 0.1]
GP_LENGTH_CHOICES = [5.0, 10.0, 20.0]
GP_NOISE_CHOICES = [1e-4, 5e-4, 1e-3]
KALMAN_PROCESS_CHOICES = [1e-4, 5e-4, 1e-3]
KALMAN_MEAS_CHOICES = [1e-3, 5e-3, 1e-2]
PCA_COMPONENT_CHOICES = [2, 3, 4, 5]


def build_smoother_specs(active_families: Sequence[str]) -> List[SmootherSpec]:
    active_families = list(active_families)

    return [
        SmootherSpec(
            name="savitzky_golay",
            param_options=[SG_ORDER_CHOICES, SG_WINDOW_CHOICES],
            build=lambda values: make_sg_params(int(values[0]), int(values[1])),
            validate=lambda params, n: validate_sg_params(params),
            apply=lambda spectra, params, wavelengths: apply_sg(spectra, params),
            complexity=lambda params: sg_complexity(params),
        ),
        SmootherSpec(
            name="wavelet",
            param_options=[active_families, WAVELET_LEVEL_CHOICES],
            build=lambda values: make_wavelet_params(values[0], int(values[1])),
            validate=lambda params, n: validate_wavelet_params(params, n),
            apply=lambda spectra, params, wavelengths: apply_wavelet(spectra, params),
            complexity=lambda params: wavelet_complexity(params),
        ),
        SmootherSpec(
            name="cascade",
            param_options=[SG_ORDER_CHOICES, SG_WINDOW_CHOICES, active_families, WAVELET_LEVEL_CHOICES, [0.0, 1.0]],
            build=lambda values: make_cascade_params(
                make_sg_params(int(values[0]), int(values[1])),
                make_wavelet_params(values[2], int(values[3])),
                float(values[4]),
            ),
            validate=lambda params, n: (
                (params.sg_params is None or validate_sg_params(params.sg_params))
                and (params.wavelet_params is None or validate_wavelet_params(params.wavelet_params, n))
            ),
            apply=lambda spectra, params, wavelengths: apply_cascade(spectra, params),
            complexity=lambda params: cascade_complexity(params),
        ),
        SmootherSpec(
            name="butterworth",
            param_options=[BUTTER_ORDER_CHOICES, BUTTER_CUTOFF_CHOICES],
            build=lambda values: ButterworthParams(order=int(values[0]), cutoff=float(values[1])),
            validate=lambda params, n: validate_butter_params(params),
            apply=lambda spectra, params, wavelengths: apply_butterworth(spectra, params),
            complexity=lambda params: butterworth_complexity(params),
        ),
        SmootherSpec(
            name="wiener",
            param_options=[WIENER_WINDOW_CHOICES, WIENER_NOISE_CHOICES],
            build=lambda values: WienerParams(window=int(values[0]), noise_power=float(values[1])),
            validate=lambda params, n: validate_wiener_params(params),
            apply=lambda spectra, params, wavelengths: apply_wiener(spectra, params),
            complexity=lambda params: wiener_complexity(params),
        ),
        SmootherSpec(
            name="median_hampel",
            param_options=[MEDIAN_KERNEL_CHOICES, HAMPEL_WINDOW_CHOICES, HAMPEL_SIGMA_CHOICES],
            build=lambda values: MedianHampelParams(kernel_size=int(values[0]), hampel_window=int(values[1]), hampel_sigma=float(values[2])),
            validate=lambda params, n: validate_median_hampel_params(params),
            apply=lambda spectra, params, wavelengths: apply_median_hampel(spectra, params),
            complexity=lambda params: median_hampel_complexity(params),
        ),
        SmootherSpec(
            name="bilateral",
            param_options=[BILATERAL_COLOR_CHOICES, BILATERAL_SPATIAL_CHOICES],
            build=lambda values: BilateralParams(sigma_color=float(values[0]), sigma_spatial=float(values[1])),
            validate=lambda params, n: validate_bilateral_params(params),
            apply=lambda spectra, params, wavelengths: apply_bilateral(spectra, params),
            complexity=lambda params: bilateral_complexity(params),
        ),
        SmootherSpec(
            name="whittaker",
            param_options=[WHITTAKER_LAM_CHOICES, WHITTAKER_D_CHOICES],
            build=lambda values: WhittakerParams(lam=float(values[0]), diff_order=int(values[1])),
            validate=lambda params, n: validate_whittaker_params(params),
            apply=lambda spectra, params, wavelengths: apply_whittaker(spectra, params),
            complexity=lambda params: whittaker_complexity(params),
        ),
        SmootherSpec(
            name="smoothing_spline",
            param_options=[SPLINE_S_CHOICES],
            build=lambda values: SplineParams(smooth_factor=float(values[0])),
            validate=lambda params, n: validate_spline_params(params),
            apply=lambda spectra, params, wavelengths: apply_spline(spectra, params, wavelengths=wavelengths),
            complexity=lambda params: spline_complexity(params),
        ),
        SmootherSpec(
            name="total_variation",
            param_options=[TV_WEIGHT_CHOICES],
            build=lambda values: TVParams(weight=float(values[0])),
            validate=lambda params, n: validate_tv_params(params),
            apply=lambda spectra, params, wavelengths: apply_tv(spectra, params),
            complexity=lambda params: tv_complexity(params),
        ),
        SmootherSpec(
            name="gaussian_process",
            param_options=[GP_LENGTH_CHOICES, GP_NOISE_CHOICES],
            build=lambda values: GPParams(length_scale=float(values[0]), noise_level=float(values[1])),
            validate=lambda params, n: validate_gp_params(params),
            apply=lambda spectra, params, wavelengths: apply_gp(spectra, params, wavelengths=wavelengths),
            complexity=lambda params: gp_complexity(params),
        ),
        SmootherSpec(
            name="kalman",
            param_options=[KALMAN_PROCESS_CHOICES, KALMAN_MEAS_CHOICES],
            build=lambda values: KalmanParams(process_var=float(values[0]), meas_var=float(values[1])),
            validate=lambda params, n: validate_kalman_params(params),
            apply=lambda spectra, params, wavelengths: apply_kalman(spectra, params),
            complexity=lambda params: kalman_complexity(params),
        ),
        SmootherSpec(
            name="pca",
            param_options=[PCA_COMPONENT_CHOICES],
            build=lambda values: PCAParams(n_components=int(values[0])),
            validate=lambda params, n: validate_pca_params(params, n),
            apply=lambda spectra, params, wavelengths: apply_pca(spectra, params),
            complexity=lambda params: pca_complexity(params),
        ),
    ]


def build_gene_space(specs: Sequence[SmootherSpec]) -> List[object]:
    max_params = max(len(spec.param_options) for spec in specs)
    spaces: List[object] = [list(range(len(specs)))]
    for _ in range(max_params):
        spaces.append({"low": 0.0, "high": 1.0})
    return spaces


def decode_parameters(spec: SmootherSpec, gene_values: Sequence[float]):
    decoded = []
    for idx, options in enumerate(spec.param_options):
        if not options:
            continue
        value = float(gene_values[idx])
        if len(options) == 1:
            decoded.append(options[0])
            continue
        scaled = np.clip(value, 0.0, 1.0 - 1e-12)
        index = int(np.floor(scaled * len(options)))
        index = min(index, len(options) - 1)
        decoded.append(options[index])
    return spec.build(decoded)


def serialize_params(obj):
    if obj is None:
        return None
    if is_dataclass(obj):
        return {key: serialize_params(value) for key, value in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_params(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def robust_snr(raw: np.ndarray, smoothed: np.ndarray, residual: np.ndarray) -> float:
    signal = np.median(np.abs(raw), axis=1)
    signal = np.maximum(signal, 1e-9)
    noise = np.median(np.abs(residual), axis=1) / 0.6745
    noise_floor = np.maximum(1e-3 * signal, 1e-9)
    noise = np.maximum(noise, noise_floor)
    ratio = signal / noise
    snr_db = 20.0 * np.log10(np.maximum(ratio, 1e-9))
    return float(np.mean(snr_db))


def high_frequency_energy(smoothed: np.ndarray, wavelengths: np.ndarray) -> float:
    centered = smoothed - smoothed.mean(axis=1, keepdims=True)
    if centered.shape[1] < 3:
        return 0.0
    delta = np.median(np.diff(wavelengths)) if len(wavelengths) > 1 else 1.0
    fft_vals = np.fft.rfft(centered, axis=1)
    freqs = np.fft.rfftfreq(centered.shape[1], d=delta)
    if freqs.size == 0:
        return 0.0
    cutoff = freqs.max() * 0.5
    mask = freqs >= cutoff
    if not mask.any():
        return 0.0
    hf = np.sum(np.abs(fft_vals[:, mask]) ** 2, axis=1)
    total = np.sum(np.abs(fft_vals) ** 2, axis=1) + 1e-9
    return float(np.mean(hf / total))


def residual_rms(residual: np.ndarray) -> float:
    return float(np.mean(np.sqrt(np.mean(residual ** 2, axis=1))))


def curvature_penalty(smoothed: np.ndarray) -> float:
    if smoothed.shape[1] < 3:
        return 0.0
    second = np.diff(smoothed, n=2, axis=1)
    return float(np.mean(np.abs(second)))


def smoothness_metrics(raw: np.ndarray, smoothed: np.ndarray, wavelengths: np.ndarray) -> Dict[str, float]:
    residual = raw - smoothed
    change = np.median(np.abs(residual), axis=1)
    signal = np.maximum(np.median(np.abs(raw), axis=1), 1e-9)
    metrics = {
        "robust_snr": robust_snr(raw, smoothed, residual),
        "high_freq_energy": high_frequency_energy(smoothed, wavelengths),
        "residual_rms": residual_rms(residual),
        "curvature_penalty": curvature_penalty(smoothed),
        "change_fraction": float(np.mean(change / (signal + 1e-9))),
    }
    return metrics


def run_ga(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    output_dir: Path,
    families: Sequence[str] | None = None,
    ga_params: Dict[str, float] | None = None,
) -> GAOutputs:
    spectra = np.asarray(spectra, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)
    if spectra.ndim != 2:
        raise ValueError("spectra must be a 2D array")

    active_families = list(families) if families else list(AVAILABLE_FAMILIES)
    if not active_families:
        raise ValueError("At least one wavelet family must be provided")

    ga_params = ga_params or {}

    smoother_specs = build_smoother_specs(active_families)
    exclude = {str(name).strip() for name in ga_params.get("exclude_smoothers", [])}
    if exclude:
        smoother_specs = [spec for spec in smoother_specs if spec.name not in exclude]
        if not smoother_specs:
            raise ValueError("All smoothers were excluded via ga.exclude_smoothers")
    smoother_names = {idx: spec.name for idx, spec in enumerate(smoother_specs)}
    gene_space = build_gene_space(smoother_specs)
    raw_high_freq = high_frequency_energy(spectra, wavelengths)

    num_generations_total = int(ga_params.get("num_generations", 100))
    sol_per_pop = int(ga_params.get("sol_per_pop", 64))
    random_seed = int(ga_params.get("random_seed", 42))
    num_parents_mating = int(ga_params.get("num_parents_mating", max(2, sol_per_pop // 2)))
    keep_parents = int(ga_params.get("keep_parents", max(1, min(num_parents_mating, sol_per_pop // 4))))

    num_parents_mating = min(num_parents_mating, sol_per_pop)
    keep_parents = min(keep_parents, num_parents_mating)

    rng = np.random.default_rng(random_seed)

    def random_population() -> np.ndarray:
        pop = []
        for _ in range(sol_per_pop):
            chromosome = [float(rng.integers(0, len(smoother_specs)))]
            if len(gene_space) > 1:
                chromosome.extend(rng.random(len(gene_space) - 1).tolist())
            pop.append(np.array(chromosome, dtype=float))
        return np.array(pop)

    total_generations = 0
    best_overall_genome = None
    best_overall_fitness = -np.inf
    best_overall_spec_idx = None
    best_overall_params = None
    best_overall_smoothed = None
    best_metrics = None
    fitness_history: List[float] = []

    population = random_population()

    def fitness_func(_ga: pygad.GA, solution: Sequence[float], _idx: int) -> float:
        which = int(round(solution[0]))
        if which < 0 or which >= len(smoother_specs):
            return -1e9
        spec = smoother_specs[which]
        param_genes = solution[1:1 + len(spec.param_options)]
        params = decode_parameters(spec, param_genes)

        if not spec.validate(params, spectra.shape[1]):
            return -1e9

        try:
            y_sm = spec.apply(spectra, params, wavelengths)
        except Exception:
            return -1e9

        if y_sm.shape != spectra.shape or not np.all(np.isfinite(y_sm)):
            return -1e9

        metrics = smoothness_metrics(spectra, y_sm, wavelengths)
        comp = spec.complexity(params)

        if not np.isfinite(metrics["robust_snr"]):
            return -1e9

        change_fraction = metrics["change_fraction"]
        change_penalty_low = max(0.002 - change_fraction, 0.0)
        change_penalty_high = max(change_fraction - 0.05, 0.0)
        hf_gain = max(raw_high_freq - metrics["high_freq_energy"], 0.0)
        fitness = (
            2.0 * metrics["robust_snr"]
            - 2.0 * metrics["high_freq_energy"]
            - 0.8 * metrics["residual_rms"]
            - 0.5 * metrics["curvature_penalty"]
            - 0.3 * comp
            + 5000.0 * hf_gain
            - 20000.0 * change_penalty_low
            - 20000.0 * change_penalty_high
        )
        return fitness

    while total_generations < num_generations_total:
        remaining_generations = num_generations_total - total_generations
        ga = pygad.GA(
            num_generations=remaining_generations,
            sol_per_pop=sol_per_pop,
            num_genes=len(gene_space),
            initial_population=population,
            parent_selection_type="tournament",
            K_tournament=4,
            num_parents_mating=num_parents_mating,
            keep_parents=keep_parents,
            crossover_type="uniform",
            crossover_probability=0.9,
            mutation_type="random",
            mutation_by_replacement=True,
            gene_space=gene_space,
            fitness_func=fitness_func,
            stop_criteria=["saturate_10"],
            random_seed=random_seed + total_generations,
        )

        ga.run()

        stage_best_genome, stage_best_fitness, _ = ga.best_solution()
        fitness_history.extend(ga.best_solutions_fitness)

        which = int(round(stage_best_genome[0]))
        spec = smoother_specs[which]
        params = decode_parameters(spec, stage_best_genome[1:1 + len(spec.param_options)])

        try:
            smoothed = spec.apply(spectra, params, wavelengths)
            metrics = smoothness_metrics(spectra, smoothed, wavelengths)
            metrics["complexity"] = spec.complexity(params)
        except Exception:
            metrics = None
            smoothed = None

        if metrics is not None and stage_best_fitness > best_overall_fitness:
            best_overall_fitness = stage_best_fitness
            best_overall_genome = stage_best_genome
            best_overall_spec_idx = which
            best_overall_params = params
            best_overall_smoothed = smoothed
            best_metrics = metrics

        total_generations += ga.generations_completed
        if total_generations >= num_generations_total:
            break

        # restart: keep elites, refill rest randomly
        fitness_values = np.array(ga.last_generation_fitness)
        elite_indices = np.argsort(fitness_values)[-keep_parents:]
        elites = ga.population[elite_indices]
        population = random_population()
        population[:len(elites)] = elites

    if best_overall_genome is None:
        raise RuntimeError("GA failed to produce a valid solution")

    smoother_name = smoother_names[best_overall_spec_idx]
    smoothed_final = best_overall_smoothed
    metrics_final = best_metrics

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(smoothed_final, columns=wavelengths).to_csv(output_dir / "smoothed_OPT.csv", index=False)

    summary_payload = {
        "genome": list(map(float, best_overall_genome)),
        "metrics": metrics_final,
        "chosen_smoother": {
            "id": best_overall_spec_idx,
            "name": smoother_name,
        },
        "parameters": serialize_params(best_overall_params),
        "families": list(active_families),
        "ga_parameters": {
            "num_generations": num_generations_total,
            "sol_per_pop": sol_per_pop,
            "num_parents_mating": num_parents_mating,
            "keep_parents": keep_parents,
            "random_seed": random_seed,
        },
    }
    (output_dir / "GA_denoise_best.json").write_text(json.dumps(summary_payload, indent=2))

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(fitness_history, marker="o")
    plt.title("GA best fitness per generation")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "ga_fitness_curve.png", dpi=200)
    plt.close()

    return GAOutputs(
        best_genome=best_overall_genome,
        fitness=float(best_overall_fitness),
        spectra_smoothed=smoothed_final,
        metrics=metrics_final,
        smoother_name=smoother_name,
        parameters=serialize_params(best_overall_params),
    )
