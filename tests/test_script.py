"""Run the ECG pipeline on a deterministic synthetic recording.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import neurokit2 as nk
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import script


SYNTHETIC_FS = 256
SYNTHETIC_DURATION_SECONDS = 60
SYNTHETIC_HEART_RATE = 72
ARTIFACT_START_SECONDS = 24
ARTIFACT_END_SECONDS = 30


def generate_synthetic_recording(file_path):
    """Generate a noisy ECG with a controlled electrode-movement artefact."""
    clean_ecg_millivolts = np.asarray(
        nk.ecg_simulate(
            duration=SYNTHETIC_DURATION_SECONDS,
            sampling_rate=SYNTHETIC_FS,
            heart_rate=SYNTHETIC_HEART_RATE,
            noise=0.02,
            random_state=42,
            method="ecgsyn",
        ),
        dtype=float,
    )

    time = np.arange(len(clean_ecg_millivolts)) / SYNTHETIC_FS
    random_generator = np.random.default_rng(2026)

    # Add baseline wander, 50-Hz interference, and broadband measurement
    # noise across the complete recording.
    noisy_ecg_millivolts = (
        clean_ecg_millivolts
        + 0.10 * np.sin(2 * np.pi * 0.25 * time)
        + 0.025 * np.sin(2 * np.pi * 50 * time)
        + 0.08 * random_generator.standard_normal(len(time))
    )

    # Add an amplitude-modulated movement burst with consecutive positive and
    # negative excursions. It obscures several QRS complexes without making
    # the signal flat.
    movement_mask = (
        (time >= ARTIFACT_START_SECONDS)
        & (time < ARTIFACT_END_SECONDS)
    )
    movement_time = time[movement_mask] - ARTIFACT_START_SECONDS
    movement_duration = ARTIFACT_END_SECONDS - ARTIFACT_START_SECONDS
    movement_envelope = (
        0.35
        + 0.65
        * np.sin(np.pi * movement_time / movement_duration) ** 2
    )
    movement_artefact = (
        movement_envelope
        * (
            0.90
            * np.sin(
                2 * np.pi * 8 * movement_time
                + 0.30 * np.sin(2 * np.pi * 0.6 * movement_time)
            )
            + 0.20 * np.sin(2 * np.pi * 13 * movement_time)
        )
        + 0.15 * np.sin(2 * np.pi * 0.35 * movement_time)
    )
    noisy_ecg_millivolts[movement_mask] = (
        movement_artefact
        + 0.05 * clean_ecg_millivolts[movement_mask]
    )

    ecg_volts = noisy_ecg_millivolts / 1000
    info = mne.create_info(
        ch_names=["ECG"],
        sfreq=SYNTHETIC_FS,
        ch_types=["ecg"],
    )
    raw = mne.io.RawArray(ecg_volts[np.newaxis, :], info, verbose="ERROR")
    raw.save(file_path, overwrite=True, verbose="ERROR")


def calculate_feature_output_summary(record):
    """Summarise morphology outputs and retain every HRV feature value."""
    summary = {
        "pipeline_version": record["pipeline_version"],
        "file_name": record["file_name"],
        "rr_interval_exclusion": record["rr_interval_exclusion"],
        "morphology": {},
        "hrv": {},
    }

    for feature_name, values in record["morphology"].items():
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            summary["morphology"][feature_name] = {
                "count": 0,
                "mean": None,
                "standard_deviation": None,
                "median": None,
                "minimum": None,
                "maximum": None,
            }
            continue

        summary["morphology"][feature_name] = {
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values)),
            "median": float(np.median(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }

    for feature_name, value in record["hrv"].items():
        if value is None or not np.isfinite(value):
            summary["hrv"][feature_name] = None
        else:
            summary["hrv"][feature_name] = float(value)

    return summary


def _plot_morphology_group(ax, morphology, names, labels, title, ylabel):
    """Plot morphology means with standard-deviation error bars."""
    available_names = [
        name for name in names if morphology[name]["mean"] is not None
    ]
    available_labels = [
        label
        for name, label in zip(names, labels)
        if name in available_names
    ]
    means = [morphology[name]["mean"] for name in available_names]
    standard_deviations = [
        morphology[name]["standard_deviation"] for name in available_names
    ]

    positions = np.arange(len(available_names))
    ax.bar(
        positions,
        means,
        yerr=standard_deviations,
        capsize=4,
        color="#4C78A8",
        alpha=0.85,
    )
    ax.set_xticks(positions, available_labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def _format_feature_value(value, missing_label="Unavailable"):
    if value is None:
        return missing_label
    return f"{value:.8g}"


def plot_feature_output_summary(summary, output_path):
    """Plot morphology summaries and the complete set of HRV values."""
    morphology = summary["morphology"]
    fig = plt.figure(figsize=(18, 17))
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 2.2])
    axes = np.array([
        [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
        [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
    ])

    _plot_morphology_group(
        axes[0, 0],
        morphology,
        [
            "qrs_duration_ms",
            "st_segment_ms",
            "pr_interval_ms",
            "pr_segment_ms",
            "qt_interval_ms",
        ],
        ["QRS", "ST", "PR interval", "PR segment", "QT"],
        "Morphology intervals: mean ± SD",
        "Milliseconds",
    )
    _plot_morphology_group(
        axes[0, 1],
        morphology,
        ["p_amp_uV", "q_amp_uV", "r_amp_uV", "s_amp_uV", "t_amp_uV"],
        ["P", "Q", "R", "S", "T"],
        "Wave amplitudes: mean ± SD",
        "Microvolts",
    )
    _plot_morphology_group(
        axes[1, 0],
        morphology,
        ["q_angle_deg", "r_angle_deg", "s_angle_deg"],
        ["Q", "R", "S"],
        "Wave angles: mean ± SD",
        "Degrees",
    )

    axes[1, 1].axis("off")
    axes[1, 1].set_title("RR-interval exclusion summary")
    exclusion = summary["rr_interval_exclusion"]
    exclusion_rows = [
        ["Detected R-peaks", exclusion["n_rpeaks_detected"]],
        [
            "RR intervals before exclusion",
            exclusion["n_rr_intervals_before"],
        ],
        ["RR intervals excluded", exclusion["n_rr_intervals_excluded"]],
        ["RR intervals retained", exclusion["n_rr_intervals_retained"]],
        ["Excluded duration (s)", f"{exclusion['excluded_duration_sec']:.3f}"],
    ]
    exclusion_table = axes[1, 1].table(
        cellText=exclusion_rows,
        colLabels=["Item", "Value"],
        cellLoc="center",
        loc="center",
    )
    exclusion_table.auto_set_font_size(False)
    exclusion_table.set_fontsize(9)
    exclusion_table.scale(1, 1.45)

    hrv_axis = fig.add_subplot(grid[2, :])
    hrv_axis.axis("off")
    hrv_axis.set_title("Complete HRV feature output", fontsize=14, pad=14)

    hrv_items = list(summary["hrv"].items())
    split = (len(hrv_items) + 1) // 2
    left_items = hrv_items[:split]
    right_items = hrv_items[split:]
    hrv_rows = []
    for row_index in range(split):
        left_name, left_value = left_items[row_index]
        if row_index < len(right_items):
            right_name, right_value = right_items[row_index]
        else:
            right_name, right_value = "", None

        hrv_rows.append([
            left_name,
            _format_feature_value(left_value),
            right_name,
            (
                _format_feature_value(right_value, missing_label="")
                if right_name
                else ""
            ),
        ])

    hrv_table = hrv_axis.table(
        cellText=hrv_rows,
        colLabels=["HRV feature", "Value", "HRV feature", "Value"],
        cellLoc="center",
        colWidths=[0.30, 0.18, 0.30, 0.18],
        loc="upper center",
    )
    hrv_table.auto_set_font_size(False)
    hrv_table.set_fontsize(9)
    hrv_table.scale(1, 1.55)

    fig.suptitle("Synthetic ECG feature-output summary", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_synthetic_demonstration():
    """Generate synthetic ECG, run script.py, and create summary outputs."""
    generated_dir = PROJECT_ROOT / "generated"
    input_dir = generated_dir / "input"
    output_dir = generated_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    recording_path = input_dir / "SYNTH001_Pre_Ictal_1_raw.fif"
    generate_synthetic_recording(recording_path)

    script.process_data(
        input_dir=input_dir,
        output_dir=output_dir,
        fs=SYNTHETIC_FS,
        min_rpeaks=50,
        min_rr_ms=300,
        max_rr_ms=2000,
        save_plots=True,
    )

    feature_json_path = output_dir / "SYNTH001_preictal_ECG_features.json"
    if not feature_json_path.exists():
        raise FileNotFoundError(
            "script.py did not create the expected feature JSON output."
        )

    with feature_json_path.open(encoding="utf-8") as stream:
        record = json.load(stream)[0]

    summary = calculate_feature_output_summary(record)
    summary_json_path = output_dir / "feature_output_summary.json"
    with summary_json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=4, allow_nan=False)

    summary_plot_path = output_dir / "feature_output_summary.png"
    plot_feature_output_summary(summary, summary_plot_path)

    result = {
        "recording": recording_path,
        "feature_json": feature_json_path,
        "ecg_plot": (
            output_dir
            / "plots"
            / "SYNTH001_Pre_Ictal_1_raw_ecg_preprocessing.png"
        ),
        "rr_plot": (
            output_dir
            / "plots"
            / "SYNTH001_Pre_Ictal_1_raw_feature_input_after_rr_exclusion.png"
        ),
        "summary_json": summary_json_path,
        "summary_plot": summary_plot_path,
        "record": record,
    }

    missing_outputs = [
        path
        for key, path in result.items()
        if key != "record" and not path.exists()
    ]
    if missing_outputs:
        missing_names = ", ".join(str(path) for path in missing_outputs)
        raise FileNotFoundError(f"Missing demonstration outputs: {missing_names}")

    return result


def display_generated_plots(result):
    """Display the three figures produced by the demonstration."""
    figures = [
        (result["ecg_plot"], "Raw and digitally filtered ECG"),
        (result["rr_plot"], "Feature input after RR-interval exclusion"),
        (result["summary_plot"], "Complete feature-output summary"),
    ]

    for image_path, title in figures:
        image = plt.imread(image_path)
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(title)
        fig.tight_layout()

    plt.show()


def main():
    print(f"Synthetic ECG demonstration using script.py v{script.__version__}")
    print(f"Demonstration file: {Path(__file__).resolve()}")
    print(f"Pipeline file:      {Path(script.__file__).resolve()}\n")

    result = run_synthetic_demonstration()
    exclusion = result["record"]["rr_interval_exclusion"]

    print("Processing completed successfully.")
    print(f"Detected R-peaks:      {exclusion['n_rpeaks_detected']}")
    print(f"Excluded RR intervals: {exclusion['n_rr_intervals_excluded']}")
    print(f"Retained RR intervals: {exclusion['n_rr_intervals_retained']}")
    print("RR handling:           exclusion only")
    print(f"Reported HRV features: {len(result['record']['hrv'])}\n")
    print("Generated files:")
    print(f"  Synthetic recording: {result['recording']}")
    print(f"  Feature JSON:         {result['feature_json']}")
    print(f"  ECG preprocessing:    {result['ecg_plot']}")
    print(f"  RR exclusion:         {result['rr_plot']}")
    print(f"  Feature summary JSON: {result['summary_json']}")
    print(f"  Feature summary plot: {result['summary_plot']}\n")
    print("Opening the three generated figures...")
    display_generated_plots(result)


if __name__ == "__main__":
    main()
