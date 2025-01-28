import os
import json
import numpy as np
import mne
import math
import neurokit2 as nk
from scipy.signal import resample
from collections import defaultdict
from neurokit2.hrv.hrv_utils import _hrv_get_rri
import matplotlib.pyplot as plt

# Disable verbose logs
mne.set_log_level('ERROR')


def ECGfilter(raw, Fs=256):
    """
    Filters and cleans the ECG signal extracted from the EDF file.
    
    Parameters:
    - raw: MNE Raw object containing the loaded data.
    - Fs: Desired sampling frequency for the ECG signal (default: 256 Hz).
    
    Returns:
    - ecg_clean: Filtered and cleaned ECG signal.
    - ecg_signal: Raw ECG signal before cleaning.
    """

    # Extract ECG channels
    if "ECG+" not in raw.ch_names or "ECG-" not in raw.ch_names:
        raise ValueError("Missing ECG+ or ECG- channels in the dataset.")
    
    ecg_plus = raw.get_data(picks="ECG+").flatten()
    ecg_minus = raw.get_data(picks="ECG-").flatten()
    ecg_signal = ecg_minus - ecg_plus

    # Resample ECG to match the target sampling frequency
    current_fs = raw.info['sfreq']
    if current_fs != Fs:
        num_samples = int(len(ecg_signal) * Fs / current_fs)
        ecg_signal = resample(ecg_signal, num_samples)

    # Clean ECG using NeuroKit2
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=Fs, method='biosppy')

    # Remove samples where amplitude does not exceed 1 µV (convert to µV for comparison)
    ecg_cleaned_muV = ecg_cleaned * 1e6
    valid_indices = np.where(np.abs(ecg_cleaned_muV) > 1)[0]
    ecg_cleaned = ecg_cleaned[valid_indices]

    # R-peak detection
    _, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=Fs, method='nabian2018')
    R = info['ECG_R_Peaks']

    # Calculate RR intervals
    RR = np.diff(R) / Fs * 1000  # RR intervals in milliseconds

    # Detect outliers using MAD
    median_RR = np.median(RR)
    mad_RR = np.median(np.abs(RR - median_RR))
    threshold = 3  # MAD threshold (tunable)
    outlier_indices = np.where(np.abs(RR - median_RR) / (mad_RR + 1e-6) > threshold)[0]

    # Remove outliers
    for idx in outlier_indices:
        a, b = R[idx], R[idx + 1]
        ecg_cleaned[a:b + 1] = np.nan

    # Remove NaN values
    ecg_clean = ecg_cleaned[~np.isnan(ecg_cleaned)]
    return ecg_clean, ecg_signal


def plot_ecg(raw_ecg, ecg_filtered, file, output_dir):
    """
    Plots and saves the raw and filtered ECG signals.
    
    Parameters:
    - raw_ecg: Raw ECG signal.
    - ecg_filtered: Filtered ECG signal.
    - file: File name for labeling the plot.
    - output_dir: Directory where the plot will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    raw_ecg_muV = raw_ecg * 1e6
    ecg_filtered_muV = ecg_filtered * 1e6

    plt.figure(figsize=(12, 6))
    plt.plot(raw_ecg_muV, label="Raw ECG (µV)", alpha=0.6, linewidth=1.0)
    plt.plot(ecg_filtered_muV, label="Filtered ECG (µV)", color='r', alpha=0.8, linewidth=1.0)
    plt.title("ECG Before and After Filtering", fontsize=14)
    plt.xlabel("Time (samples)", fontsize=12)
    plt.ylabel("Amplitude (µV)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.4)

    plt.tight_layout()

    output_image_path = os.path.join(output_dir, f"{file}_ecg_plot.png")
    plt.savefig(output_image_path, dpi=300)
    plt.close()

def angle_of_vectors(a,b,c,d):
    
     dotProduct = a*c + b*d
         # for three dimensional simply add dotProduct = a*c + b*d  + e*f 
     modOfVector1 = math.sqrt(a*a + b*b)*math.sqrt(c*c + d*d) 
         # for three dimensional simply add modOfVector = math.sqrt( a*a + b*b + e*e)*math.sqrt(c*c + d*d +f*f) 
     angle = dotProduct/modOfVector1
     angleInDegree = math.degrees(math.acos(angle))
     return angleInDegree

def morphology(sig, rpeaks, Fs):
    """
    Extracts morphological features such as QRS duration, ST segment, and angles.
    
    Parameters:
    - sig: Cleaned ECG signal.
    - rpeaks: Dictionary containing R-peak indices.
    - Fs: Sampling frequency of the signal.
    
    Returns:
    - ECG_morphology: Dictionary of extracted morphological features.
    """
    _, waves = nk.ecg_delineate(sig, rpeaks, sampling_rate = Fs)
    
    # morphological features
    qrs_dur = []
    st_seg = []
    pr_int = []
    pr_seg = []
    qt_int = []
    p_peaks = []
    q_peaks = []
    r_peaks = []
    s_peaks = []
    t_peaks = []
    q_angle = []
    r_angle = []
    s_angle = []
    for i in range(len(rpeaks['ECG_R_Peaks'])):
        p_off = waves['ECG_P_Offsets'][i]
        p_on = waves['ECG_P_Onsets'][i]
        p_peak = waves['ECG_P_Peaks'][i]
        q_peak = waves['ECG_Q_Peaks'][i]
        r_off = waves['ECG_R_Offsets'][i]
        r_on = waves['ECG_R_Onsets'][i]
        s_peak = waves['ECG_S_Peaks'][i] 
        t_off = waves['ECG_T_Offsets'][i]
        t_on = waves['ECG_T_Onsets'][i]
        t_peak = waves['ECG_T_Peaks'][i]
        if ~np.isnan([p_off,p_on,p_peak,q_peak,r_off,r_on,s_peak,t_off,t_on,t_peak]).any() and ~np.isinf([p_off,p_on,p_peak,q_peak,r_off,r_on,s_peak,t_off,t_on,t_peak]).any():
            # intervals        
            qrs = r_off - r_on            
            qrs_dur.append(qrs)
            st = t_on - r_off
            st_seg.append(st)
            pr = r_on - p_on
            pr_int.append(pr)
            qt = t_off - r_on
            qt_int.append(qt)
            pr = r_on - p_off
            pr_seg.append(pr)
            # amplitudes
            p = sig[p_peak]
            p_peaks.append(p)
            q = sig[q_peak]
            q_peaks.append(q)
            r = sig[rpeaks['ECG_R_Peaks'][i]]
            r_peaks.append(r)
            s = sig[s_peak]
            s_peaks.append(s)
            t = sig[t_peak]
            t_peaks.append(t)
            # angles
            q_x0 = q_peak
            q_y0 = sig[q_x0]
            q_x1 =  p_peak - q_x0
            q_y1 = sig[q_x1] - q_y0
            q_x2 =  rpeaks['ECG_R_Peaks'][i] - q_x0
            q_y2 = sig[q_x2] - q_y0
            angle = angle_of_vectors(q_x1,q_y1,q_x2,q_y2)
            q_angle.append(angle)    
            r_x0 = rpeaks['ECG_R_Peaks'][i]
            r_y0 = sig[r_x0]
            r_x1 =  q_peak - r_x0
            r_y1 = sig[r_x1] - r_y0
            r_x2 =  s_peak - r_x0
            r_y2 = sig[r_x2] - r_y0
            angle = angle_of_vectors(r_x1,r_y1,r_x2,r_y2)
            r_angle.append(angle)    
            s_x0 = s_peak
            s_y0 = sig[s_x0]
            s_x1 =  rpeaks['ECG_R_Peaks'][i] - s_x0
            s_y1 = sig[s_x1] - s_y0
            s_x2 =  t_peak - s_x0
            s_y2 = sig[s_x2] - s_y0
            angle = angle_of_vectors(s_x1,s_y1,s_x2,s_y2)
            s_angle.append(angle)
        
    ECG_morphology = {'qrs_dur':qrs_dur,'st_seg':st_seg,'pr_int':pr_int,
                      'pr_seg':pr_seg,'qt_int':qt_int,'p_peak':p_peaks,
                      'q_peak':q_peaks,'r_peak':r_peaks,'s_peak':s_peaks,
                      't_peak':t_peaks,'q_angle':q_angle,'r_angle':r_angle,
                      's_angle':s_angle}
    return ECG_morphology

def hrv(peaks,rpeaks,Fs):
    """
    Extracts heart rate variability (HRV) features including time-domain, frequency-domain, and non-linear metrics.
    
    Parameters:
    - peaks: Detected peaks in the ECG signal.
    - rpeaks: Dictionary containing R-peak indices.
    - Fs: Sampling frequency of the signal.
    
    Returns:
    - ECG_hrv: Dictionary of extracted HRV features.
    """
    # HRV features
    interpolation_rate = 5
    rri = _hrv_get_rri(rpeaks['ECG_R_Peaks'], sampling_rate=Fs, interpolate=True, interpolation_rate=interpolation_rate)[0]
    rri = nk.intervals_to_peaks(rri)
    hrv_time_domain = nk.hrv_time(rri, sampling_rate=Fs)
    hrv_frequency_domain = nk.hrv_frequency(peaks, sampling_rate=Fs,interpolation_rate=interpolation_rate)
    hrv_poincare = nk.hrv_nonlinear(rri, sampling_rate=Fs)
    # k_max, _ = nk.complexity_k(rri) 
    hfd, _ = nk.fractal_higuchi(rri, k_max='default')      
    d, _ = nk.complexity_dimension(rri)
    t, _ = nk.complexity_tolerance(rri,method='nolds',dimension=d)
    SampEn, _ = nk.entropy_sample(rri,dimension=d,tolerance=t)
    MSEn, _ = nk.entropy_multiscale(rri,dimension=d,tolerance=t,scale='default')
    LZC, _ = nk.complexity_lempelziv(rri,dimension=d)
    
    ECG_hrv = {'MeanNN':hrv_time_domain['HRV_MeanNN'][0],'SDNN':hrv_time_domain['HRV_SDNN'][0],
                'RMSSD':hrv_time_domain['HRV_RMSSD'][0],'SDSD':hrv_time_domain['HRV_SDSD'][0],
                'CVNN':hrv_time_domain['HRV_CVNN'][0],'CVSD':hrv_time_domain['HRV_CVSD'][0],
                'MedianNN':hrv_time_domain['HRV_MedianNN'][0],'MadNN':hrv_time_domain['HRV_MadNN'][0],
                'MCVNN':hrv_time_domain['HRV_MCVNN'][0],'IQRNN':hrv_time_domain['HRV_IQRNN'][0],
                'Prc20NN':hrv_time_domain['HRV_Prc20NN'][0],'Prc80NN':hrv_time_domain['HRV_Prc80NN'][0],
                'MinNN':hrv_time_domain['HRV_MinNN'][0],'MaxNN':hrv_time_domain['HRV_MaxNN'][0],
                'LF':hrv_frequency_domain['HRV_LF'][0],'HF':hrv_frequency_domain['HRV_HF'][0],
                'LFHF':hrv_frequency_domain['HRV_LFHF'][0],'SD1':hrv_poincare['HRV_SD1'][0],
                'SD2':hrv_poincare['HRV_SD2'][0],'SD1SD2':hrv_poincare['HRV_SD1SD2'][0],
                'CSI':hrv_poincare['HRV_CSI'][0],'CVI':hrv_poincare['HRV_CVI'][0],
                'SampEn':SampEn,'MSEn':MSEn,'HFD':hfd,'LZC':LZC}
    return ECG_hrv

# Main Processing Pipeline
def process_data(edf_directory, output_dir):
    """
    Main pipeline to process ECG data, extract HRV and morphological features, and save them.
    
    Parameters:
    - edf_directory: Directory containing the EDF files.
    - output_dir: Directory where the processed features will be saved.
    """
    edf_files = [f for f in os.listdir(edf_directory) if f.endswith('.edf')]
    
    # Identify files and group by patient
    patient_files = defaultdict(list)
    
    for file in edf_files:
        # Extract patient identifier and condition
        parts = file.split('_')
        patient_id = parts[0]  # e.g., "2016M33"
        patient_files[patient_id].append(file)
        
    # Filter patients with both preictal and interictal files
    valid_patients = {
        patient: files
        for patient, files in patient_files.items()
        if any("Pre" in f for f in files) and any("Inter" in f for f in files)
    }
    
    for patient, files in valid_patients.items():
        print(f"Processing patient: {patient}")
        preictal_ecg = []
        interictal_ecg = []
    
        for file in files:
            file_path = os.path.join(edf_directory, file)
            raw = mne.io.read_raw_edf(file_path, preload=True)
    
            try:
                ecg_cleaned, ecg_raw = ECGfilter(raw, Fs=256)
            except ValueError as e:
                print(f"Skipping file {file} due to missing ECG channels: {e}")
                continue
    
            peaks, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=256, method='nabian2018')
            if np.count_nonzero(peaks) <= 50:
                print(f"Skipping file {file} due to insufficient R-peaks ({np.count_nonzero(peaks)}).")
                continue
    
            # plot_ecg(ecg_raw, ecg_cleaned, file, output_dir)
            
            # Extract ECG morphological features
            morphology_features = morphology(ecg_cleaned, rpeaks, Fs=256)
            
            # Extract ECG HRV features
            hrv_features = hrv(peaks, rpeaks, Fs=256)
            
            # Combine ECG features
            ecg_features = {
                "morphology": morphology_features,
                "hrv": hrv_features
            }
            
            # Store features based on condition
            if "Pre" in file:
                preictal_ecg.append(ecg_features)
            elif "Inter" in file:
                interictal_ecg.append(ecg_features)
        
        # Save data for preictal condition
        if preictal_ecg:
            with open(os.path.join(output_dir, f"{patient}_preictal_ECG_features.json"), "w") as f:
                json.dump(preictal_ecg, f, indent=4)
            print(f"Saved preictal ECG features for patient {patient}.")
        
        # Save data for interictal condition
        if interictal_ecg:
            with open(os.path.join(output_dir, f"{patient}_interictal_ECG_features.json"), "w") as f:
                json.dump(interictal_ecg, f, indent=4)
            print(f"Saved interictal ECG features for patient {patient}.")

# Define directories and run the pipeline
edf_directory = "C:\\Path\\To\\EDF_Files"
output_dir = "C:\\Path\\To\\Processed_Data"
process_data(edf_directory, output_dir)
