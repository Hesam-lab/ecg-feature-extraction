# ECG Feature Extraction from EDF/FIF Recordings

This repository contains a Python pipeline for extracting ECG-derived features from clinical electrophysiological recordings. The code was developed for processing ECG signals stored in **EDF** or **FIF** files and extracting both **heart rate variability (HRV)** and **ECG morphology** features.

The pipeline uses **MNE-Python** for reading electrophysiological recordings and **NeuroKit2** for ECG cleaning, R-peak detection, HRV analysis, and ECG waveform delineation.

---

## Associated Publication

This repository provides a reusable implementation of the ECG ingestion, preprocessing, quality-control, and feature-extraction workflow developed from the research code used in:

> H. Shokouh Alaei et al., "Preictal Reduction in Heart Rate Variability Entropy Is Associated with Functional/Dissociative Seizures and Provides Modest Discrimination from Epileptic Seizures," *Epilepsy & Behavior*, vol. 183, 111181, 2026. https://doi.org/10.1016/j.yebeh.2026.111181

The repository covers preprocessing and feature extraction. It does not contain the restricted clinical dataset or the complete statistical and machine-learning analysis used in the publication.

> **RR handling:** This repository uses the **RR-interval exclusion method** applied in the associated *Epilepsy & Behavior* paper. Abnormal RR intervals are removed from HRV estimation, and their bounding detected beats are excluded from morphology estimation. The pipeline does not reconstruct excluded intervals or generate synthetic R-peaks.

---

## Overview

The script performs the following steps:

1. Loads EDF/FIF files using MNE.
2. Detects the ECG/EKG channel automatically.
3. Resamples the ECG signal to a target sampling frequency.
4. Cleans the ECG signal using NeuroKit2.
5. Detects R-peaks.
6. Identifies abnormal RR intervals using MAD and configurable physiological limits.
7. Excludes abnormal RR observations from HRV estimation using the paper method.
8. Excludes detected beats bounding rejected intervals from morphology analysis.
9. Extracts HRV and ECG morphology features from the selected observations.
10. Saves extracted features as JSON files.

---

## Signal Preprocessing and RR-Interval Handling

### ECG resampling and digital filtering

The ECG channel is loaded in its original physical units. If its sampling frequency differs from the requested `--fs` value, the signal is first resampled with SciPy's polyphase method (`resample_poly`). The resampled, unfiltered signal is retained as the raw ECG for comparison in the preprocessing plot.

Digital filtering is then performed with `neurokit2.ecg_clean(..., method="biosppy")`. In NeuroKit2 0.2.10, this applies:

- a zero-phase finite impulse response (FIR) band-pass filter from **0.67 to 45 Hz** using forward-backward filtering;
- a filter length of `1.5 × sampling frequency`, increased by one when necessary to produce an odd number of taps (385 taps at 256 Hz); and
- removal of the remaining DC offset by subtracting the filtered signal mean.

The resulting digitally filtered ECG—not the raw ECG—is used for Nabian R-peak detection and waveform delineation.

### Abnormal RR identification and the paper's exclusion method

R-peaks are detected once with NeuroKit2's `nabian2018` method. The detected peak array is preserved: the pipeline does not relocate or overwrite these observed peaks. RR intervals are calculated as the time differences between consecutive detected R-peaks.

Each RR interval receives the following modified median absolute deviation (MAD) score:

```text
modified MAD score = 0.6745 × |RR − median(RR)| / MAD(RR)
```

An RR interval is excluded when at least one of these conditions is met:

- its modified MAD score is greater than `--mad_threshold` (default `3.5`);
- it is shorter than `--min_rr_ms` (default `300 ms`); or
- it is longer than `--max_rr_ms` (default `2000 ms`).

The diagnostic figure shows both the physiological limits and the RR limits corresponding to the MAD threshold. The MAD-derived limits are calculated as:

```text
median(RR) ± mad_threshold × MAD(RR) / 0.6745
```

This **exclusion method is the method used in the associated paper and the only RR-gap method implemented by the pipeline**. For morphology estimation, both detected R-peaks bounding an excluded RR interval are omitted from waveform delineation. For HRV estimation, only retained RR values are passed to NeuroKit2 and the custom entropy/complexity calculations. No synthetic RR values or beat timestamps are generated.

### Gaps in the retained RR series

The diagnostic plot preserves the original recording time and leaves blank gaps where RR intervals were excluded. For calculation, however, the retained RR values are concatenated into one array, matching the paper's workflow. Consequently, the last retained RR value before an excluded region and the first retained value after it are treated as adjacent observations. This artificial adjacency can influence successive-difference measures such as RMSSD and SDSD, nonlinear measures, entropy measures, and interpolated frequency-domain features. MeanNN, MinNN, and MaxNN depend only on the retained values and are not affected by that adjacency.

Interpolation is applied only where a uniformly sampled RR tachogram is required for a particular HRV calculation:

| Feature group | RR-series processing |
|---|---|
| MeanNN, SDNN, RMSSD, SDSD, MinNN and MaxNN | Calculated directly from the retained, non-interpolated RR values |
| LF, HF and LF/HF | NeuroKit2 constructs cumulative time from the concatenated retained RR values and applies its default quadratic interpolation at **4 Hz** before Welch spectral estimation |
| SD1, SD2, SD1/SD2, CSI and CVI | Calculated directly from the retained, non-interpolated RR values |
| Sample, fuzzy, Rényi, permutation and dispersion entropy | Calculated directly from the retained, non-interpolated RR values |
| Spectral entropy, Higuchi fractal dimension and Lempel–Ziv complexity | The concatenated retained RR values are linearly interpolated at **4 Hz** by the pipeline's `interpolate_rr()` function |

These interpolation steps create an evenly sampled representation of the retained RR series; they do **not** reconstruct excluded RR intervals or fill the original artefact gap. Because original RR timestamps are not supplied to the HRV functions, the interpolation operates on the shortened, concatenated timeline.

---

## Data Format

The code is designed to work with EDF/FIF files named using a patient identifier followed by the recording condition.

Example file names:

```text
SubID_Pre_Ictal_1.edf
SubID_Inter_Ictal_1.edf
```

The patient ID is extracted from the first part of the file name before the first underscore.

For example:

```text
100_Pre_Ictal_1.edf  ->  patient ID: 100
```

The condition is detected from the file name:

| Filename pattern | Detected condition |
|---|---|
| `Pre_Ictal`, `Pre-Ictal`, `Preictal` | `preictal` |
| `Inter_Ictal`, `Inter-Ictal`, `Interictal` | `interictal` |
| `Post_Ictal`, `Post-Ictal`, `Postictal` | `postictal` |
| `Ictal` | `ictal` |

If your file naming convention is different, update the `parse_condition()` function in the script.

---

## Features Extracted

### HRV Features

The pipeline extracts time-domain, frequency-domain, and nonlinear HRV features, including:

- MeanNN
- SDNN
- RMSSD
- SDSD
- MinNN
- MaxNN
- LF
- HF
- LF/HF
- SD1
- SD2
- SD1/SD2
- CSI
- CVI
- Sample entropy
- Fuzzy entropy
- Rényi entropy
- Permutation entropy
- Dispersion entropy
- Spectral entropy
- Higuchi fractal dimension
- Lempel-Ziv complexity

### ECG Morphology Features

The pipeline also extracts morphology-related ECG features, including:

- QRS duration
- ST segment
- PR interval
- PR segment
- QT interval
- P-wave amplitude
- Q-wave amplitude
- R-wave amplitude
- S-wave amplitude
- T-wave amplitude
- Q angle
- R angle
- S angle

Interval features are saved in milliseconds, and amplitude features are saved in microvolts.

---

## Requirements

The repository was developed using:

```text
mne==1.8.0
neurokit2==0.2.10
```

Additional dependencies are listed in `requirements.txt`.

The pinned environment has been tested with Python 3.12.

## Installation

Clone the repository:

```bash
git clone https://github.com/Hesam-lab/ecg-feature-extraction.git
cd ecg-feature-extraction
```

Creating a virtual environment is recommended.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the included `script.py` file:

```bash
python script.py \
  --input_dir "path/to/edf_or_fif_files" \
  --output_dir "path/to/output_folder" \
  --fs 256
```

On Windows PowerShell, the command can be entered on one line:

```powershell
python script.py --input_dir "C:\path\to\recordings" --output_dir "C:\path\to\output" --fs 256
```

### Required arguments

| Argument | Description |
|---|---|
| `--input_dir` | Directory containing EDF or FIF recordings |
| `--output_dir` | Directory in which JSON feature files will be saved |

### Optional arguments

| Argument | Default | Description |
|---|---:|---|
| `--fs` | `256` | Target ECG sampling frequency in hertz |
| `--save_plots` | Off | Save ECG and RR-interval quality-control plots |
| `--min_rpeaks` | `50` | Minimum detected R-peak count and corresponding retained RR count |
| `--mad_threshold` | `3.5` | Maximum modified MAD z-score: `0.6745 × |RR − median RR| / MAD` |
| `--min_rr_ms` | `300` | Minimum plausible RR interval in milliseconds |
| `--max_rr_ms` | `2000` | Maximum plausible RR interval in milliseconds |

To save ECG plots:

```bash
python script.py \
  --input_dir "path/to/edf_or_fif_files" \
  --output_dir "path/to/output_folder" \
  --fs 256 \
  --save_plots
```

To view all command-line options:

```bash
python script.py --help
```

## Output

For each patient and condition, the pipeline saves a JSON file containing the extracted ECG features.

Example output files:

```text
SubID_preictal_ECG_features.json
SubID_interictal_ECG_features.json
```

Each JSON file contains:

- file name
- patient ID
- condition
- recording duration
- sampling rate
- number of detected R-peaks and retained RR/morphology observations
- ECG cleaning information
- RR-interval exclusion information, including rejected intervals, reasons, affected segments, and retained summaries
- HRV features
- ECG morphology features

### Synthetic demonstration

The demonstration script generates a deterministic 60-second noisy ECG using NeuroKit2. It adds baseline wander, 50-Hz interference, broadband measurement noise, and a controlled six-second electrode-movement burst with alternating high and low amplitudes. The burst obscures consecutive QRS complexes and produces abnormal RR intervals without exposing clinical data.

The generated RR diagnostic keeps all panels on the original 0–60-second time axis. Excluded intervals are marked in the middle panel, and the bottom panel shows only retained RR observations used for HRV estimation. Excluded regions remain blank and are not filled or joined across the original recording timeline.

## Run the Synthetic Demonstration

The repository includes a simple demonstration script rather than a unit-test suite. Run it directly to generate the synthetic ECG, process it through `script.py`, retain the outputs, and display the figures:

```bash
python tests/test_script.py
```

This direct command creates an ignored root-level `generated/` directory containing:

- the complete pipeline JSON output;
- an ECG preprocessing plot comparing the raw noisy signal with the digitally filtered signal (BioSPPy 0.67–45 Hz FIR band-pass and DC-offset removal);
- a feature-input plot showing retained/excluded detected peaks, shaded excluded ECG segments, and the RR values used for feature estimation;
- `feature_output_summary.json`, containing morphology summary statistics and every HRV feature value; and
- `feature_output_summary.png`, containing morphology summaries, exclusion information, and a complete HRV feature table.

## Data Privacy and Responsible Use

- No clinical recordings or real participant data are included in this repository.
- Use pseudonymous identifiers in input filenames; the identifier and source filename are written to the JSON output.
- Keep recordings, derived features, plots, and other potentially sensitive outputs outside version control.
- This is research software, not a medical device. Validate the signal units, preprocessing choices, quality-control limits, and extracted features for your own acquisition system and study protocol.

---

## Citation

```text
Alaei, H. (2026). ECG Feature Extraction from EDF/FIF Recordings. GitHub repository:
https://github.com/Hesam-lab/ecg-feature-extraction
```
