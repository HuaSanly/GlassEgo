import csv
import json
import math
import os
import tempfile
from pathlib import Path

import allantools
import matplotlib.pyplot as plt
import numpy as np


IMU_COLUMNS = ("sensor_type", "sequence", "timestamp_ns", "x", "y", "z")
SENSOR_TYPES = ("gyroscope", "accelerometer")
AXIS_NAMES = ("x", "y", "z")
MIN_DURATION_SECONDS = 60.0
RECOMMENDED_DURATION_SECONDS = 3.0 * 60.0 * 60.0
WHITE_NOISE_MAX_TAU_SECONDS = 10.0
WHITE_NOISE_REFERENCE_TAU_SECONDS = 1.0
RANDOM_WALK_REFERENCE_TAU_SECONDS = 3.0


def calibrate_imu(
    unit_dir: str | Path,
    report_path: str | Path,
    plot_path: str | Path,
) -> tuple[dict, dict]:
    """从静止 IMU 数据估计 Kalibr 连续时间噪声参数。"""
    unit_dir = Path(unit_dir)
    streams = _load_imu_streams(unit_dir / "imu.csv")
    analyses = {
        sensor_type: _analyze_stream(sensor_type, streams[sensor_type])
        for sensor_type in SENSOR_TYPES
    }

    warnings = []
    for sensor_type, analysis in analyses.items():
        if analysis["duration_seconds"] < RECOMMENDED_DURATION_SECONDS:
            warnings.append(
                f"{sensor_type} duration is shorter than the recommended 3 hours: "
                f"{analysis['duration_seconds']:.1f} s"
            )

    gyro = analyses["gyroscope"]
    accel = analyses["accelerometer"]
    imu_config = {
        "gyro_noise_density": gyro["white_noise"]["maximum"],
        "gyro_random_walk": gyro["random_walk"]["maximum"],
        "accel_noise_density": accel["white_noise"]["maximum"],
        "accel_random_walk": accel["random_walk"]["maximum"],
    }
    if not all(
        math.isfinite(value) and value > 0.0 for value in imu_config.values()
    ):
        raise RuntimeError("Allan analysis returned invalid IMU noise parameters")

    report = {
        "warnings": warnings,
        "gyroscope": gyro,
        "accelerometer": accel,
        "final_parameters": imu_config,
        "units": {
            "gyro_noise_density": "rad/s/sqrt(Hz)",
            "gyro_random_walk": "rad/s^2/sqrt(Hz)",
            "accel_noise_density": "m/s^2/sqrt(Hz)",
            "accel_random_walk": "m/s^3/sqrt(Hz)",
        },
    }
    _save_allan_plot(plot_path, analyses)
    _write_json(report_path, report)
    return imu_config, report


def _load_imu_streams(path: Path) -> dict[str, list[dict]]:
    if not path.is_file():
        raise FileNotFoundError(f"imu.csv not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        missing = [column for column in IMU_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(f"imu.csv is missing columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("imu.csv contains no samples")

    streams = {sensor_type: [] for sensor_type in SENSOR_TYPES}
    for row_index, row in enumerate(rows):
        raw_sensor_type = row.get("sensor_type")
        sensor_type = raw_sensor_type.strip().lower() if raw_sensor_type else ""
        if sensor_type not in streams:
            raise ValueError(
                f"Unknown sensor_type at row {row_index}: {sensor_type!r}"
            )
        try:
            sequence = int(row["sequence"])
            timestamp_ns = int(row["timestamp_ns"])
            values = np.array(
                [float(row[axis]) for axis in AXIS_NAMES],
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid imu.csv value at row {row_index}") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite imu.csv value at row {row_index}")
        streams[sensor_type].append(
            {
                "sequence": sequence,
                "timestamp_ns": timestamp_ns,
                "values": values,
            }
        )

    for sensor_type, samples in streams.items():
        if len(samples) < 2:
            raise ValueError(f"imu.csv contains fewer than two {sensor_type} samples")
        timestamps = np.array(
            [sample["timestamp_ns"] for sample in samples],
            dtype=np.int64,
        )
        sequences = np.array(
            [sample["sequence"] for sample in samples],
            dtype=np.int64,
        )
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"{sensor_type} timestamps must be strictly increasing")
        if np.any(np.diff(sequences) <= 0):
            raise ValueError(f"{sensor_type} sequences must be strictly increasing")
    return streams


def _analyze_stream(sensor_type: str, samples: list[dict]) -> dict:
    timestamps_ns = np.array(
        [sample["timestamp_ns"] for sample in samples],
        dtype=np.int64,
    )
    values = np.stack([sample["values"] for sample in samples])
    sequences = np.array(
        [sample["sequence"] for sample in samples],
        dtype=np.int64,
    )
    intervals_seconds = np.diff(timestamps_ns).astype(np.float64) * 1e-9
    median_interval = float(np.median(intervals_seconds))
    if not math.isfinite(median_interval) or median_interval <= 0.0:
        raise ValueError(f"{sensor_type} has an invalid median sample interval")
    sample_rate_hz = 1.0 / median_interval
    duration_seconds = float((timestamps_ns[-1] - timestamps_ns[0]) * 1e-9)
    if duration_seconds < MIN_DURATION_SECONDS:
        raise ValueError(
            f"{sensor_type} duration must be at least {MIN_DURATION_SECONDS:.0f} "
            f"seconds: {duration_seconds:.3f} s"
        )

    sequence_gaps = np.diff(sequences) - 1
    missing_samples = int(np.sum(sequence_gaps[sequence_gaps > 0]))
    expected_samples = len(samples) + missing_samples
    gap_threshold = 1.5 * median_interval
    timestamp_gap_count = int(np.count_nonzero(intervals_seconds > gap_threshold))

    axis_results = {}
    white_noise_values = []
    random_walk_values = []
    for axis_index, axis_name in enumerate(AXIS_NAMES):
        taus, deviations, errors, counts = allantools.oadev(
            values[:, axis_index],
            rate=sample_rate_hz,
            data_type="freq",
            taus="decade",
        )
        taus = np.asarray(taus, dtype=np.float64)
        deviations = np.asarray(deviations, dtype=np.float64)
        errors = np.asarray(errors, dtype=np.float64)
        counts = np.asarray(counts, dtype=np.float64)
        valid = (
            np.isfinite(taus)
            & np.isfinite(deviations)
            & (taus > 0.0)
            & (deviations > 0.0)
        )
        taus = taus[valid]
        deviations = deviations[valid]
        errors = errors[valid]
        counts = counts[valid]
        if len(taus) < 3:
            raise RuntimeError(
                f"AllanTools returned too few valid points for {sensor_type}.{axis_name}"
            )

        white_noise, white_intercept = _fit_fixed_slope(
            taus,
            deviations,
            slope=-0.5,
            reference_tau=WHITE_NOISE_REFERENCE_TAU_SECONDS,
            maximum_tau=WHITE_NOISE_MAX_TAU_SECONDS,
        )
        random_walk, random_intercept = _fit_fixed_slope(
            taus,
            deviations,
            slope=0.5,
            reference_tau=RANDOM_WALK_REFERENCE_TAU_SECONDS,
        )
        white_noise_values.append(white_noise)
        random_walk_values.append(random_walk)
        axis_results[axis_name] = {
            "tau_seconds": taus.tolist(),
            "allan_deviation": deviations.tolist(),
            "allan_error": errors.tolist(),
            "sample_pairs": [int(value) for value in counts],
            "white_noise": white_noise,
            "random_walk": random_walk,
            "white_fit_intercept": white_intercept,
            "random_walk_fit_intercept": random_intercept,
        }

    return {
        "sample_count": len(samples),
        "duration_seconds": duration_seconds,
        "sample_rate_hz": sample_rate_hz,
        "median_interval_seconds": median_interval,
        "interval_std_seconds": float(np.std(intervals_seconds)),
        "relative_interval_jitter": float(
            np.std(intervals_seconds) / median_interval
        ),
        "missing_sequence_count": missing_samples,
        "missing_sequence_rate": (
            float(missing_samples / expected_samples) if expected_samples else 0.0
        ),
        "timestamp_gap_count": timestamp_gap_count,
        "axes": axis_results,
        "white_noise": _summarize_axes(white_noise_values),
        "random_walk": _summarize_axes(random_walk_values),
    }


def _fit_fixed_slope(
    taus: np.ndarray,
    deviations: np.ndarray,
    slope: float,
    reference_tau: float,
    maximum_tau: float | None = None,
) -> tuple[float, float]:
    mask = np.ones(len(taus), dtype=bool)
    if maximum_tau is not None:
        mask &= taus < maximum_tau
    fit_taus = taus[mask]
    fit_deviations = deviations[mask]
    if len(fit_taus) < 2:
        raise RuntimeError(
            f"Not enough Allan points to fit slope {slope:+.1f}"
        )
    log_intercept = float(
        np.mean(np.log(fit_deviations) - slope * np.log(fit_taus))
    )
    intercept = math.exp(log_intercept)
    value = intercept * reference_tau**slope
    return float(value), float(intercept)


def _summarize_axes(values: list[float]) -> dict:
    return {
        "per_axis": {
            axis_name: float(value)
            for axis_name, value in zip(AXIS_NAMES, values)
        },
        "mean": float(np.mean(values)),
        "maximum": float(np.max(values)),
    }


def _save_allan_plot(path: str | Path, analyses: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    titles = {
        "gyroscope": "Gyroscope Allan deviation",
        "accelerometer": "Accelerometer Allan deviation",
    }
    for axis, sensor_type in zip(axes, SENSOR_TYPES):
        analysis = analyses[sensor_type]
        for axis_name in AXIS_NAMES:
            result = analysis["axes"][axis_name]
            taus = np.asarray(result["tau_seconds"], dtype=np.float64)
            deviations = np.asarray(result["allan_deviation"], dtype=np.float64)
            line = axis.loglog(taus, deviations, marker=".", label=axis_name)[0]
            color = line.get_color()
            white_fit = result["white_fit_intercept"] * taus**-0.5
            random_walk_fit = result["random_walk_fit_intercept"] * taus**0.5
            axis.loglog(taus, white_fit, linestyle="--", color=color, alpha=0.35)
            axis.loglog(
                taus,
                random_walk_fit,
                linestyle=":",
                color=color,
                alpha=0.35,
            )
        axis.set_xlabel("Tau (s)")
        axis.set_ylabel("Allan deviation")
        axis.set_title(titles[sensor_type])
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    figure.tight_layout()
    _save_figure_atomic(figure, path)
    plt.close(figure)


def _save_figure_atomic(figure, path: Path) -> None:
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}_",
        suffix=path.suffix,
        dir=path.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        figure.savefig(temporary_path, dpi=160)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}_",
        suffix=path.suffix,
        dir=path.parent,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.write("\n")
        Path(temporary_name).replace(path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
