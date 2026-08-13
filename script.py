"""ECG feature-extraction pipeline with R-peak quality-control diagnostics."""

import argparse
import json
import logging
import math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import matplotlib.pyplot as plt
import mne
import neurokit2 as nk
import numpy as np
import scipy.interpolate
from scipy.signal import resample_poly
from tqdm import tqdm

__version__ = "1.1.0"

mne.set_log_level("ERROR")

FS = 256
MIN_RPEAKS = 50
MAD_THRESHOLD = 3.5
MAD_SCALE = 0.6745
MIN_RR_MS = 300.0
MAX_RR_MS = 2000.0

logger = logging.getLogger(__name__)


def json_friendly(x):
    """Convert numpy values to normal Python values before saving as JSON."""
    if isinstance(x, dict):
        return {k: json_friendly(v) for k, v in x.items()}

    if isinstance(x, list):
        return [json_friendly(v) for v in x]

    if isinstance(x, tuple):
        return [json_friendly(v) for v in x]

    if isinstance(x, np.ndarray):
        return json_friendly(x.tolist())

    if isinstance(x, np.integer):
        return int(x)

    if isinstance(x, np.floating):
        x = float(x)
        return None if math.isnan(x) or math.isinf(x) else x

    if isinstance(x, float):
        return None if math.isnan(x) or math.isinf(x) else x

    return x


def get_ecg_channel(raw):
    """Find the ECG channel in an MNE Raw object."""
    ecg_channels = [
        ch for ch in raw.ch_names
        if "ecg" in ch.lower() or "ekg" in ch.lower()
    ]

    if len(ecg_channels) == 0:
        raise ValueError(f"No ECG channel found. Available channels: {raw.ch_names}")

    return ecg_channels[0]


def clean_ecg(raw, fs=FS):
    """Extract, resample, and clean ECG."""
    if fs <= 0:
        raise ValueError("Target sampling frequency must be positive.")

    ecg_channel = get_ecg_channel(raw)
    ecg = raw.get_data(picks=[ecg_channel]).flatten()

    original_fs = float(raw.info["sfreq"])
    if original_fs <= 0:
        raise ValueError("Recording sampling frequency must be positive.")

    if not np.isclose(original_fs, fs):
        ratio = Fraction(float(fs) / original_fs).limit_denominator(1000)
        ecg = resample_poly(ecg, up=ratio.numerator, down=ratio.denominator)

    ecg_clean = nk.ecg_clean(ecg, sampling_rate=fs, method="biosppy")

    info = {
        "ecg_channel": ecg_channel,
        "original_fs": float(original_fs),
        "target_fs": fs,
        "n_samples": len(ecg_clean),
        "cleaning_method": "neurokit2_biosppy",
        "digital_filter": "FIR band-pass",
        "passband_hz": [0.67, 45.0],
        "filter_order": int(1.5 * fs) + (1 - int(1.5 * fs) % 2),
        "dc_offset_removed": True,
    }

    return ecg_clean, ecg, info


def detect_rpeaks(ecg_clean, fs=FS):
    """Detect R-peaks using NeuroKit2."""
    _, rpeaks = nk.ecg_peaks(
        ecg_clean,
        sampling_rate=fs,
        method="nabian2018"
    )

    rpeaks["ECG_R_Peaks"] = np.asarray(rpeaks["ECG_R_Peaks"], dtype=int)
    rpeaks["sampling_rate"] = fs

    return rpeaks


def _rr_summary(rr_ms):
    """Summarise RR interval values for quality-control reporting."""
    rr_ms = np.asarray(rr_ms, dtype=float)
    if len(rr_ms) == 0:
        return {
            "minimum_ms": None,
            "median_ms": None,
            "maximum_ms": None,
        }

    return {
        "minimum_ms": float(np.min(rr_ms)),
        "median_ms": float(np.median(rr_ms)),
        "maximum_ms": float(np.max(rr_ms)),
    }


def exclude_abnormal_rr_intervals(
    rpeaks,
    fs=FS,
    mad_threshold=MAD_THRESHOLD,
    min_rr_ms=MIN_RR_MS,
    max_rr_ms=MAX_RR_MS,
):
    """Identify and exclude abnormal RR intervals without changing R-peaks.

    R-peaks detected by the Nabian et al method are preserved exactly. An RR
    interval is rejected if it violates the physiological limits or if its
    modified MAD z-score exceeds ``mad_threshold``. Beats bounding rejected
    intervals are excluded from morphology analysis, while the retained RR
    observations are supplied directly to HRV analysis. No peak is inserted,
    removed, or moved.
    """
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")
    if mad_threshold <= 0:
        raise ValueError("mad_threshold must be positive.")
    if min_rr_ms <= 0 or max_rr_ms <= 0:
        raise ValueError("RR interval limits must be positive.")
    if min_rr_ms >= max_rr_ms:
        raise ValueError("min_rr_ms must be smaller than max_rr_ms.")

    detected = np.asarray(rpeaks["ECG_R_Peaks"], dtype=int)
    if len(detected) < 2:
        raise ValueError("At least two detected R-peaks are required.")
    if np.any(np.diff(detected) <= 0):
        raise ValueError("Detected R-peaks must be strictly increasing.")

    rr_ms = np.diff(detected) / fs * 1000
    rr_time_sec = detected[1:] / fs
    median_rr_ms = float(np.median(rr_ms))
    mad_rr_ms = float(np.median(np.abs(rr_ms - median_rr_ms)))

    if np.isclose(mad_rr_ms, 0):
        mad_scores = np.zeros(len(rr_ms), dtype=float)
        mad_outliers = np.zeros(len(rr_ms), dtype=bool)
        mad_lower_rr_ms = None
        mad_upper_rr_ms = None
    else:
        mad_scores = (
            MAD_SCALE * np.abs(rr_ms - median_rr_ms) / mad_rr_ms
        )
        mad_outliers = mad_scores > mad_threshold
        mad_limit_delta_ms = mad_threshold * mad_rr_ms / MAD_SCALE
        mad_lower_rr_ms = median_rr_ms - mad_limit_delta_ms
        mad_upper_rr_ms = median_rr_ms + mad_limit_delta_ms

    short_intervals = rr_ms < min_rr_ms
    long_intervals = rr_ms > max_rr_ms
    excluded_mask = mad_outliers | short_intervals | long_intervals
    retained_mask = ~excluded_mask

    morphology_peak_mask = np.ones(len(detected), dtype=bool)
    excluded_indices = np.flatnonzero(excluded_mask)
    morphology_peak_mask[excluded_indices] = False
    morphology_peak_mask[excluded_indices + 1] = False

    retained_rr_ms = rr_ms[retained_mask]
    retained_rr_time_sec = rr_time_sec[retained_mask]
    morphology_peaks = detected[morphology_peak_mask]

    excluded_intervals = []
    for interval_index in excluded_indices:
        reasons = []
        if short_intervals[interval_index]:
            reasons.append("below_minimum")
        if long_intervals[interval_index]:
            reasons.append("above_maximum")
        if mad_outliers[interval_index]:
            reasons.append("mad_outlier")

        excluded_intervals.append({
            "interval_index": int(interval_index),
            "start_peak_sample": int(detected[interval_index]),
            "end_peak_sample": int(detected[interval_index + 1]),
            "start_time_sec": float(detected[interval_index] / fs),
            "end_time_sec": float(detected[interval_index + 1] / fs),
            "rr_ms": float(rr_ms[interval_index]),
            "mad_score": float(mad_scores[interval_index]),
            "reasons": reasons,
        })

    report = {
        "method": "abnormal_rr_interval_exclusion",
        "rpeak_detector": "nabian2018",
        "rpeaks_modified": False,
        "mad_threshold": float(mad_threshold),
        "mad_score_definition": f"{MAD_SCALE} * abs(RR - median_RR) / MAD",
        "minimum_rr_ms": float(min_rr_ms),
        "maximum_rr_ms": float(max_rr_ms),
        "median_rr_ms": median_rr_ms,
        "mad_rr_ms": mad_rr_ms,
        "mad_lower_rr_ms": mad_lower_rr_ms,
        "mad_upper_rr_ms": mad_upper_rr_ms,
        "n_rpeaks_detected": int(len(detected)),
        "n_morphology_peaks_retained": int(len(morphology_peaks)),
        "n_rr_intervals_before": int(len(rr_ms)),
        "n_rr_intervals_retained": int(len(retained_rr_ms)),
        "n_rr_intervals_excluded": int(np.sum(excluded_mask)),
        "n_mad_outliers": int(np.sum(mad_outliers)),
        "n_short_intervals": int(np.sum(short_intervals)),
        "n_long_intervals": int(np.sum(long_intervals)),
        "excluded_percentage": float(np.mean(excluded_mask) * 100),
        "excluded_duration_sec": float(np.sum(rr_ms[excluded_mask]) / 1000),
        "excluded_interval_indices": excluded_indices.tolist(),
        "excluded_intervals": excluded_intervals,
        "excluded_morphology_peak_samples": detected[~morphology_peak_mask].tolist(),
        "rr_before": _rr_summary(rr_ms),
        "rr_retained": _rr_summary(retained_rr_ms),
        "retained_rr_handling": "concatenated_after_exclusion",
        "retained_rr_caveat": (
            "The last retained RR value before an excluded region and the "
            "first retained RR value after it are treated as adjacent for "
            "successive-difference, nonlinear, and interpolated HRV metrics."
        ),
    }

    morphology_rpeaks = dict(rpeaks)
    morphology_rpeaks["ECG_R_Peaks"] = morphology_peaks
    morphology_rpeaks["sampling_rate"] = fs

    exclusion_result = {
        "detected_rpeaks": detected,
        "rr_ms": rr_ms,
        "rr_time_sec": rr_time_sec,
        "retained_mask": retained_mask,
        "retained_rr_ms": retained_rr_ms,
        "retained_rr_time_sec": retained_rr_time_sec,
        "morphology_peak_mask": morphology_peak_mask,
        "morphology_rpeaks": morphology_rpeaks,
    }

    return exclusion_result, report


def valid_index(x, n):
    try:
        if x is None:
            return False

        x = float(x)

        if np.isnan(x) or np.isinf(x):
            return False

        x = int(round(x))
        return 0 <= x < n

    except Exception:
        return False


def to_index(x):
    return int(round(float(x)))


def get_wave(waves, name, i):
    try:
        return waves[name][i]
    except Exception:
        return np.nan


def angle_between_points(left, centre, right, signal, fs=FS):
    """
    Angle at the centre point.

    The x-axis is time in ms and the y-axis is ECG amplitude in microvolts.
    """
    signal_uv = signal * 1e6

    p1 = np.array([left / fs * 1000, signal_uv[left]])
    p2 = np.array([centre / fs * 1000, signal_uv[centre]])
    p3 = np.array([right / fs * 1000, signal_uv[right]])

    v1 = p1 - p2
    v2 = p3 - p2

    denom = np.linalg.norm(v1) * np.linalg.norm(v2)

    if denom == 0:
        return np.nan

    cos_angle = np.dot(v1, v2) / denom
    cos_angle = np.clip(cos_angle, -1, 1)

    return float(np.degrees(np.arccos(cos_angle)))


def extract_morphology(ecg_clean, rpeaks, fs=FS):
    """Extract ECG morphology features from delineated ECG waves."""
    features = {
        "qrs_duration_ms": [],
        "st_segment_ms": [],
        "pr_interval_ms": [],
        "pr_segment_ms": [],
        "qt_interval_ms": [],
        "p_amp_uV": [],
        "q_amp_uV": [],
        "r_amp_uV": [],
        "s_amp_uV": [],
        "t_amp_uV": [],
        "q_angle_deg": [],
        "r_angle_deg": [],
        "s_angle_deg": [],
    }

    r = np.asarray(rpeaks["ECG_R_Peaks"], dtype=int)

    if len(r) == 0:
        return features

    try:
        _, waves = nk.ecg_delineate(
            ecg_clean,
            rpeaks,
            sampling_rate=fs,
            method="dwt"
        )
    except Exception as e:
        logger.warning("ECG delineation failed: %s", e)
        return features

    n = len(ecg_clean)

    for i, r_peak in enumerate(r):
        p_on = get_wave(waves, "ECG_P_Onsets", i)
        p_peak = get_wave(waves, "ECG_P_Peaks", i)
        p_off = get_wave(waves, "ECG_P_Offsets", i)

        q_peak = get_wave(waves, "ECG_Q_Peaks", i)

        r_on = get_wave(waves, "ECG_R_Onsets", i)
        r_off = get_wave(waves, "ECG_R_Offsets", i)

        s_peak = get_wave(waves, "ECG_S_Peaks", i)

        t_on = get_wave(waves, "ECG_T_Onsets", i)
        t_peak = get_wave(waves, "ECG_T_Peaks", i)
        t_off = get_wave(waves, "ECG_T_Offsets", i)

        points = [
            p_on, p_peak, p_off,
            q_peak, r_on, r_peak, r_off,
            s_peak, t_on, t_peak, t_off,
        ]

        if not all(valid_index(x, n) for x in points):
            continue

        p_on = to_index(p_on)
        p_peak = to_index(p_peak)
        p_off = to_index(p_off)
        q_peak = to_index(q_peak)
        r_on = to_index(r_on)
        r_peak = to_index(r_peak)
        r_off = to_index(r_off)
        s_peak = to_index(s_peak)
        t_on = to_index(t_on)
        t_peak = to_index(t_peak)
        t_off = to_index(t_off)

        qrs_ms = (r_off - r_on) / fs * 1000
        st_ms = (t_on - r_off) / fs * 1000
        pr_interval_ms = (r_on - p_on) / fs * 1000
        pr_segment_ms = (r_on - p_off) / fs * 1000
        qt_ms = (t_off - r_on) / fs * 1000

        # Skip clearly impossible delineations.
        if any(x <= 0 for x in [qrs_ms, pr_interval_ms, qt_ms]):
            continue

        features["qrs_duration_ms"].append(float(qrs_ms))
        features["st_segment_ms"].append(float(st_ms))
        features["pr_interval_ms"].append(float(pr_interval_ms))
        features["pr_segment_ms"].append(float(pr_segment_ms))
        features["qt_interval_ms"].append(float(qt_ms))

        features["p_amp_uV"].append(float(ecg_clean[p_peak] * 1e6))
        features["q_amp_uV"].append(float(ecg_clean[q_peak] * 1e6))
        features["r_amp_uV"].append(float(ecg_clean[r_peak] * 1e6))
        features["s_amp_uV"].append(float(ecg_clean[s_peak] * 1e6))
        features["t_amp_uV"].append(float(ecg_clean[t_peak] * 1e6))

        features["q_angle_deg"].append(
            angle_between_points(p_peak, q_peak, r_peak, ecg_clean, fs)
        )

        features["r_angle_deg"].append(
            angle_between_points(q_peak, r_peak, s_peak, ecg_clean, fs)
        )

        features["s_angle_deg"].append(
            angle_between_points(r_peak, s_peak, t_peak, ecg_clean, fs)
        )

    return features


def get_df_value(df, col):
    try:
        return float(df[col].iloc[0])
    except Exception:
        return np.nan


def extract_hrv(rr_intervals_ms, fs=FS):
    """Extract HRV features from the RR series selected for analysis."""
    rr_ms = np.asarray(rr_intervals_ms, dtype=float)
    rr_ms = rr_ms[np.isfinite(rr_ms) & (rr_ms > 0)]

    features = {
        "MeanNN": np.nan,
        "SDNN": np.nan,
        "RMSSD": np.nan,
        "SDSD": np.nan,
        "MinNN": np.nan,
        "MaxNN": np.nan,
        "LF": np.nan,
        "HF": np.nan,
        "LFHF": np.nan,
        "SD1": np.nan,
        "SD2": np.nan,
        "SD1SD2": np.nan,
        "CSI": np.nan,
        "CVI": np.nan,
        "SpecEn": np.nan,
        "RenEn_alpha2": np.nan,
        "LZC": np.nan,
    }

    for m in [1, 2]:
        for r_frac in [0.1, 0.2]:
            features[f"SampEn_m{m}_r{r_frac}"] = np.nan
            features[f"FuzzyEn_m{m}_r{r_frac}"] = np.nan

    for m in [3, 4, 5]:
        features[f"PermEn_m{m}"] = np.nan

    for c in [5, 6, 7]:
        features[f"DispEn_c{c}"] = np.nan

    for kmax in [5, 10, 20]:
        features[f"HFD_k{kmax}"] = np.nan

    if len(rr_ms) < 3:
        return features

    # Passing retained RRI values directly prevents NeuroKit2 from
    # reconstructing or modifying the detected R-peak locations.
    hrv_input = {
        "RRI": rr_ms,
        "sampling_rate": fs,
    }
    rr = rr_ms / 1000
    rr_std = np.std(rr)

    try:
        hrv_time = nk.hrv_time(hrv_input, sampling_rate=fs, show=False)

        features["MeanNN"] = get_df_value(hrv_time, "HRV_MeanNN")
        features["SDNN"] = get_df_value(hrv_time, "HRV_SDNN")
        features["RMSSD"] = get_df_value(hrv_time, "HRV_RMSSD")
        features["SDSD"] = get_df_value(hrv_time, "HRV_SDSD")
        features["MinNN"] = get_df_value(hrv_time, "HRV_MinNN")
        features["MaxNN"] = get_df_value(hrv_time, "HRV_MaxNN")

    except Exception as e:
        logger.warning("Time-domain HRV failed: %s", e)

    try:
        hrv_freq = nk.hrv_frequency(
            hrv_input,
            sampling_rate=fs,
            interpolation_rate=4,
            psd_method="welch",
            normalize=True,
            show=False
        )

        features["LF"] = get_df_value(hrv_freq, "HRV_LF")
        features["HF"] = get_df_value(hrv_freq, "HRV_HF")
        features["LFHF"] = get_df_value(hrv_freq, "HRV_LFHF")

    except Exception as e:
        logger.warning("Frequency-domain HRV failed: %s", e)

    try:
        hrv_non = nk.hrv_nonlinear(hrv_input, sampling_rate=fs, show=False)

        features["SD1"] = get_df_value(hrv_non, "HRV_SD1")
        features["SD2"] = get_df_value(hrv_non, "HRV_SD2")
        features["SD1SD2"] = get_df_value(hrv_non, "HRV_SD1SD2")
        features["CSI"] = get_df_value(hrv_non, "HRV_CSI")
        features["CVI"] = get_df_value(hrv_non, "HRV_CVI")

    except Exception as e:
        logger.warning("Nonlinear HRV failed: %s", e)

    for m in [1, 2]:
        for r_frac in [0.1, 0.2]:
            tolerance = r_frac * rr_std

            try:
                value, _ = nk.entropy_sample(
                    rr,
                    delay=1,
                    dimension=m,
                    tolerance=tolerance
                )
                features[f"SampEn_m{m}_r{r_frac}"] = float(value)
            except Exception:
                pass

            try:
                value, _ = nk.entropy_fuzzy(
                    rr,
                    delay=1,
                    dimension=m,
                    tolerance=tolerance,
                    n=2
                )
                features[f"FuzzyEn_m{m}_r{r_frac}"] = float(value)
            except Exception:
                pass

    try:
        value, _ = nk.entropy_renyi(rr, alpha=2)
        features["RenEn_alpha2"] = float(value)
    except Exception:
        pass

    for m in [3, 4, 5]:
        try:
            value, _ = nk.entropy_permutation(
                rr,
                delay=1,
                dimension=m
            )
            features[f"PermEn_m{m}"] = float(value)
        except Exception:
            pass

    for c in [5, 6, 7]:
        try:
            value, _ = nk.entropy_dispersion(
                rr,
                delay=1,
                dimension=2,
                c=c
            )
            features[f"DispEn_c{c}"] = float(value)
        except Exception:
            pass

    rr_interp = interpolate_rr(rr)

    if rr_interp is not None:
        try:
            psd = nk.signal_psd(rr_interp, sampling_rate=4)
            power = psd["Power"].values.astype(float)

            if power.sum() > 0:
                power = power / power.sum()
                spen, _ = nk.entropy_shannon(freq=power)
                features["SpecEn"] = float(spen / np.log2(len(power)))

        except Exception:
            pass

        for kmax in [5, 10, 20]:
            try:
                value, _ = nk.fractal_higuchi(rr_interp, k_max=kmax)
                features[f"HFD_k{kmax}"] = float(value)
            except Exception:
                pass

        try:
            value, _ = nk.complexity_lempelziv(
                rr_interp,
                symbolize="mean"
            )
            features["LZC"] = float(value)
        except Exception:
            pass

    return features


def interpolate_rr(rr, target_fs=4):
    """Interpolate RR intervals to a regular time axis."""
    if len(rr) < 3:
        return None

    rr_time = np.cumsum(rr)

    if rr_time[-1] <= rr_time[0]:
        return None

    n_samples = int((rr_time[-1] - rr_time[0]) * target_fs)

    if n_samples < 3:
        return None

    new_time = np.linspace(rr_time[0], rr_time[-1], n_samples)

    try:
        f = scipy.interpolate.interp1d(
            rr_time,
            rr,
            kind="linear",
            fill_value="extrapolate"
        )

        return f(new_time)

    except Exception:
        return None


def plot_ecg_preprocessing(raw_ecg, clean_ecg, fs, file_name, output_dir):
    """Plot the noisy input ECG and digitally filtered ECG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = min(len(raw_ecg), len(clean_ecg))
    t = np.arange(n) / fs

    plt.figure(figsize=(12, 5))
    plt.plot(t, raw_ecg[:n] * 1e6, label="Raw noisy ECG", alpha=0.6)
    plt.plot(
        t,
        clean_ecg[:n] * 1e6,
        label="Digitally filtered ECG (0.67–45 Hz FIR; DC removed)",
        alpha=0.8,
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.title("ECG preprocessing")
    plt.legend()
    plt.tight_layout()

    out_file = output_dir / f"{Path(file_name).stem}_ecg_preprocessing.png"
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_feature_input_after_rr_exclusion(
    clean_ecg,
    rr_exclusion,
    report,
    fs,
    file_name,
    output_dir,
):
    """Plot excluded observations and the retained HRV RR-series input."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    signal = np.asarray(clean_ecg, dtype=float)
    detected = np.asarray(rr_exclusion["detected_rpeaks"], dtype=int)
    retained_rr_mask = np.asarray(rr_exclusion["retained_mask"], dtype=bool)
    morphology_peak_mask = np.asarray(
        rr_exclusion["morphology_peak_mask"],
        dtype=bool,
    )
    rr_ms = np.asarray(rr_exclusion["rr_ms"], dtype=float)
    rr_time_sec = np.asarray(rr_exclusion["rr_time_sec"], dtype=float)
    excluded_indices = np.flatnonzero(~retained_rr_mask)
    recording_duration_sec = len(signal) / fs
    start = 0
    stop = len(signal)
    time = np.arange(start, stop) / fs
    visible_mask = (detected >= start) & (detected < stop)
    retained_visible = detected[visible_mask & morphology_peak_mask]
    excluded_visible = detected[visible_mask & ~morphology_peak_mask]

    fig, axes = plt.subplots(3, 1, figsize=(13, 11))

    axes[0].plot(time, signal[start:stop] * 1e6, color="0.35", linewidth=1)
    for position, interval_index in enumerate(excluded_indices):
        interval_start = detected[interval_index] / fs
        interval_end = detected[interval_index + 1] / fs
        if interval_end < start / fs or interval_start > stop / fs:
            continue
        axes[0].axvspan(
            interval_start,
            interval_end,
            color="#E45756",
            alpha=0.14,
            label="Excluded RR segment" if position == 0 else None,
        )

    axes[0].scatter(
        retained_visible / fs,
        signal[retained_visible] * 1e6,
        marker="o",
        facecolors="none",
        edgecolors="#2CA02C",
        s=48,
        label="Detected peak retained for morphology",
        zorder=3,
    )
    axes[0].scatter(
        excluded_visible / fs,
        signal[excluded_visible] * 1e6,
        marker="x",
        color="#E45756",
        s=50,
        label="Detected peak excluded from morphology",
        zorder=4,
    )
    axes[0].set_title("Nabian R-peaks and ECG segments excluded from analysis")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude (µV)")
    axes[0].set_xlim(0, recording_duration_sec)
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.2)

    axes[1].plot(
        rr_time_sec,
        rr_ms,
        "o-",
        color="0.55",
        markersize=3,
        linewidth=1,
        label="All detected RR intervals",
    )
    axes[1].scatter(
        rr_time_sec[retained_rr_mask],
        rr_ms[retained_rr_mask],
        color="#2CA02C",
        s=24,
        label="Retained",
        zorder=3,
    )
    axes[1].scatter(
        rr_time_sec[~retained_rr_mask],
        rr_ms[~retained_rr_mask],
        color="#E45756",
        marker="x",
        s=55,
        label="Excluded",
        zorder=4,
    )
    axes[1].axhline(
        report["minimum_rr_ms"],
        color="0.35",
        linestyle="--",
        linewidth=1,
        label=(
            f"Physiological limits ({report['minimum_rr_ms']:g}–"
            f"{report['maximum_rr_ms']:g} ms)"
        ),
    )
    axes[1].axhline(
        report["maximum_rr_ms"],
        color="0.35",
        linestyle="--",
        linewidth=1,
    )
    if report["mad_lower_rr_ms"] is not None:
        axes[1].axhline(
            report["mad_lower_rr_ms"],
            color="#4C78A8",
            linestyle="-.",
            linewidth=1.2,
            label=(
                "MAD limits "
                f"({report['mad_lower_rr_ms']:.1f}–"
                f"{report['mad_upper_rr_ms']:.1f} ms; "
                f"threshold {report['mad_threshold']:g})"
            ),
        )
        axes[1].axhline(
            report["mad_upper_rr_ms"],
            color="#4C78A8",
            linestyle="-.",
            linewidth=1.2,
        )
    axes[1].set_title("RR-interval series before exclusion")
    axes[1].set_xlabel("Time of second detected R-peak (s)")
    axes[1].set_ylabel("RR interval (ms)")
    axes[1].set_xlim(0, recording_duration_sec)
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.2)

    retained_rr_plot = np.where(retained_rr_mask, rr_ms, np.nan)
    axes[2].plot(
        rr_time_sec,
        retained_rr_plot,
        "o-",
        color="#2CA02C",
        markersize=3,
        linewidth=1,
        label="Retained RR used for HRV",
    )
    axes[2].set_title(
        "RR values used for HRV estimation "
        f"({report['n_rr_intervals_excluded']} excluded; gaps left blank)"
    )
    axes[2].legend(loc="best")
    axes[2].set_xlabel("Time of second detected R-peak (s)")
    axes[2].set_ylabel("RR interval (ms)")
    axes[2].set_xlim(0, recording_duration_sec)
    axes[2].grid(alpha=0.2)

    fig.suptitle(
        f"Feature-estimation input after RR-interval exclusion: {file_name}",
        fontsize=14,
    )
    fig.tight_layout()

    out_file = (
        output_dir
        / f"{Path(file_name).stem}_feature_input_after_rr_exclusion.png"
    )
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_condition(file_name):
    """
    Detect condition from file name.

    examples:
    SubID_Pre_Ictal_1.edf
    SubID_Inter_Ictal_1.edf
    """
    name = file_name.lower()

    if "pre_ictal" in name or "preictal" in name or "pre-ictal" in name:
        return "preictal"

    if "post_ictal" in name or "postictal" in name or "post-ictal" in name:
        return "postictal"

    if "inter_ictal" in name or "interictal" in name or "inter-ictal" in name:
        return "interictal"

    if "ictal" in name:
        return "ictal"

    return None


def get_patient_id(file_name):
    """
    Extract patient ID from file name.

    """
    return file_name.split("_")[0]


def get_data_files(input_dir):
    input_dir = Path(input_dir)
    supported_suffixes = (".edf", ".fif", ".fif.gz")
    return sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.name.lower().endswith(supported_suffixes)
    )


def load_raw_file(file_path):
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".edf":
        return mne.io.read_raw_edf(file_path, preload=True, verbose="ERROR")

    if suffix == ".fif" or file_path.name.lower().endswith(".fif.gz"):
        return mne.io.read_raw_fif(file_path, preload=True, verbose="ERROR")

    raise ValueError(f"Unsupported file type: {file_path}")


def group_by_patient(files):
    patient_files = defaultdict(list)

    for file_path in files:
        patient_id = get_patient_id(file_path.name)
        patient_files[patient_id].append(file_path)

    return patient_files


def process_file(
    file_path,
    fs=FS,
    min_rpeaks=MIN_RPEAKS,
    mad_threshold=MAD_THRESHOLD,
    min_rr_ms=MIN_RR_MS,
    max_rr_ms=MAX_RR_MS,
    save_plot=False,
    plot_dir=None,
):
    """Process one FIF/EDF file and return ECG features."""
    raw = load_raw_file(file_path)

    ecg_clean, ecg_raw, clean_info = clean_ecg(raw, fs=fs)

    if save_plot and plot_dir is not None:
        plot_ecg_preprocessing(
            ecg_raw,
            ecg_clean,
            fs,
            file_path.name,
            plot_dir,
        )

    detected_rpeaks = detect_rpeaks(ecg_clean, fs=fs)
    rr_exclusion, rr_exclusion_report = exclude_abnormal_rr_intervals(
        detected_rpeaks,
        fs=fs,
        mad_threshold=mad_threshold,
        min_rr_ms=min_rr_ms,
        max_rr_ms=max_rr_ms,
    )
    if save_plot and plot_dir is not None:
        plot_feature_input_after_rr_exclusion(
            ecg_clean,
            rr_exclusion,
            rr_exclusion_report,
            fs,
            file_path.name,
            plot_dir,
        )

    n_rpeaks = len(rr_exclusion["detected_rpeaks"])
    n_retained_rr = len(rr_exclusion["retained_rr_ms"])
    n_morphology_rpeaks = len(
        rr_exclusion["morphology_rpeaks"]["ECG_R_Peaks"]
    )

    if n_rpeaks < min_rpeaks:
        raise ValueError(
            f"Too few R-peaks detected: {n_rpeaks} "
            f"(minimum required: {min_rpeaks})"
        )
    if n_retained_rr < min_rpeaks - 1:
        raise ValueError(
            f"Too few RR intervals retained after exclusion: {n_retained_rr} "
            f"(minimum required: {min_rpeaks - 1})"
        )

    morphology = extract_morphology(
        ecg_clean,
        rr_exclusion["morphology_rpeaks"],
        fs=fs,
    )
    hrv = extract_hrv(rr_exclusion["retained_rr_ms"], fs=fs)

    features = {
        "pipeline_version": __version__,
        "file_name": file_path.name,
        "duration_sec": len(ecg_clean) / fs,
        "sampling_rate": fs,
        "num_rpeaks_detected": n_rpeaks,
        "num_rr_intervals_retained": n_retained_rr,
        "num_rr_intervals_used_for_hrv": n_retained_rr,
        "num_morphology_rpeaks_retained": n_morphology_rpeaks,
        "cleaning": clean_info,
        "rr_interval_exclusion": rr_exclusion_report,
        "morphology": morphology,
        "hrv": hrv,
    }

    return features


def process_data(
    input_dir,
    output_dir,
    fs=FS,
    min_rpeaks=MIN_RPEAKS,
    mad_threshold=MAD_THRESHOLD,
    min_rr_ms=MIN_RR_MS,
    max_rr_ms=MAX_RR_MS,
    save_plots=False,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if fs <= 0:
        raise ValueError("Target sampling frequency must be positive.")
    if min_rpeaks < 4:
        raise ValueError("min_rpeaks must be at least 4.")
    if mad_threshold <= 0:
        raise ValueError("mad_threshold must be positive.")
    if min_rr_ms <= 0 or max_rr_ms <= 0 or min_rr_ms >= max_rr_ms:
        raise ValueError(
            "RR interval limits must be positive and min_rr_ms must be "
            "smaller than max_rr_ms."
        )
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_dir = output_dir / "plots" if save_plots else None

    data_files = get_data_files(input_dir)
    if len(data_files) == 0:
        raise FileNotFoundError(
            f"No EDF or FIF recordings found in input directory: {input_dir}"
        )

    logger.info("Found %d EDF/FIF files", len(data_files))

    patient_files = group_by_patient(data_files)

    logger.info("Found %d patients", len(patient_files))

    for patient, files in tqdm(patient_files.items(), desc="Processing patients"):
        patient_features = defaultdict(list)

        for file_path in files:
            condition = parse_condition(file_path.name)

            if condition is None:
                logger.warning("Could not detect condition for %s", file_path.name)
                continue

            try:
                features = process_file(
                    file_path,
                    fs=fs,
                    min_rpeaks=min_rpeaks,
                    mad_threshold=mad_threshold,
                    min_rr_ms=min_rr_ms,
                    max_rr_ms=max_rr_ms,
                    save_plot=save_plots,
                    plot_dir=plot_dir
                )

                features["patient_id"] = patient
                features["condition"] = condition

                patient_features[condition].append(features)

            except Exception as e:
                logger.warning("Skipping %s: %s", file_path.name, e)

        for condition, features in patient_features.items():
            if len(features) == 0:
                continue

            out_file = output_dir / f"{patient}_{condition}_ECG_features.json"

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(
                    json_friendly(features),
                    f,
                    indent=4,
                    ensure_ascii=False,
                    allow_nan=False
                )

            logger.info("Saved %s", out_file)


def main():
    parser = argparse.ArgumentParser(
        description="Extract ECG morphology and HRV features from EDF/FIF recordings."
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Folder containing EDF or FIF recordings."
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Folder where feature files will be saved."
    )

    parser.add_argument(
        "--fs",
        type=int,
        default=FS,
        help="Target sampling frequency. Default is 256 Hz."
    )

    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Save ECG and RR-interval quality-control plots."
    )

    parser.add_argument(
        "--min_rpeaks",
        type=int,
        default=MIN_RPEAKS,
        help=(
            "Minimum detected R-peak count and corresponding retained RR "
            f"count. Default is {MIN_RPEAKS}."
        ),
    )

    parser.add_argument(
        "--mad_threshold",
        type=float,
        default=MAD_THRESHOLD,
        help=(
            "Exclude RR intervals whose modified MAD z-score exceeds this "
            f"value. Default is {MAD_THRESHOLD:g}."
        ),
    )

    parser.add_argument(
        "--min_rr_ms",
        type=float,
        default=MIN_RR_MS,
        help=f"Minimum plausible RR interval in milliseconds. Default is {MIN_RR_MS:g}."
    )

    parser.add_argument(
        "--max_rr_ms",
        type=float,
        default=MAX_RR_MS,
        help=f"Maximum plausible RR interval in milliseconds. Default is {MAX_RR_MS:g}."
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    process_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fs=args.fs,
        min_rpeaks=args.min_rpeaks,
        mad_threshold=args.mad_threshold,
        min_rr_ms=args.min_rr_ms,
        max_rr_ms=args.max_rr_ms,
        save_plots=args.save_plots
    )


if __name__ == "__main__":
    main()
