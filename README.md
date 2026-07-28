# ECG Feature Extraction from EDF/FIF Recordings

This repository contains a Python pipeline for extracting ECG-derived features from clinical electrophysiological recordings. The code was developed for processing ECG signals stored in **EDF** or **FIF** files and extracting both **heart rate variability (HRV)** and **ECG morphology** features.

The pipeline uses **MNE-Python** for reading electrophysiological recordings and **NeuroKit2** for ECG cleaning, R-peak detection, HRV analysis, and ECG waveform delineation.

---

## Overview

The script performs the following steps:

1. Loads EDF/FIF files using MNE.
2. Detects the ECG/EKG channel automatically.
3. Resamples the ECG signal to a target sampling frequency.
4. Cleans the ECG signal using NeuroKit2.
5. Detects R-peaks.
6. Removes suspicious R-peaks based on RR interval outliers.
7. Extracts HRV features.
8. Extracts ECG morphology features.
9. Saves extracted features as JSON files.

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

On Windows PowerShell.

### Required arguments

| Argument | Description |
|---|---|
| `--input_dir` | Directory containing EDF or FIF recordings |
| `--output_dir` | Directory in which JSON feature files will be saved |

### Optional arguments

| Argument | Default | Description |
|---|---:|---|
| `--fs` | `256` | Target ECG sampling frequency in hertz |
| `--save_plots` | Off | Save raw-versus-cleaned ECG plots |

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
- number of detected R-peaks
- ECG cleaning information
- R-peak filtering information
- HRV features
- ECG morphology features

---

## Citation

```text
Alaei, H. (2026). ECG Feature Extraction from EDF/FIF Recordings. GitHub repository:
https://github.com/Hesam-lab/ecg-feature-extraction

