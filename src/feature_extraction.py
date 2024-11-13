# feature_extraction.py

import numpy as np
from scipy.stats import entropy
from scipy.fft import fft
from scipy.signal import find_peaks

def extract_features_ecg(segment, fs):
    """
    Extrae características de un segmento de señal ECG.

    Args:
        segment (array): Segmento de señal ECG.
        fs (int): Frecuencia de muestreo.

    Returns:
        dict: Diccionario con las características extraídas.
    """
    features = {}

    # Detección de picos R
    peaks, _ = find_peaks(segment, distance=fs*0.6)
    rr_intervals = np.diff(peaks) / fs

    if len(rr_intervals) > 0:
        heart_rate = 60 / np.mean(rr_intervals)
        hrv = np.std(rr_intervals)
    else:
        heart_rate = 0
        hrv = 0

    features['heart_rate'] = heart_rate
    features['hrv'] = hrv

    # Estadísticas básicas
    features['ecg_mean'] = np.mean(segment)
    features['ecg_std'] = np.std(segment)
    features['ecg_max'] = np.max(segment)
    features['ecg_min'] = np.min(segment)

    # Entropía espectral
    fft_vals = fft(segment)
    fft_power = np.abs(fft_vals)
    fft_power_norm = fft_power / np.sum(fft_power)
    features['ecg_entropy'] = entropy(fft_power_norm)

    return features

def extract_features_ppg(segment, fs):
    """
    Extrae características de un segmento de señal PPG.

    Args:
        segment (array): Segmento de señal PPG.
        fs (int): Frecuencia de muestreo.

    Returns:
        dict: Diccionario con las características extraídas.
    """
    features = {}

    # Detección de picos (ondas sistólicas)
    peaks, _ = find_peaks(segment, distance=fs*0.5)
    peak_intervals = np.diff(peaks) / fs

    if len(peak_intervals) > 0:
        heart_rate = 60 / np.mean(peak_intervals)
        hrv = np.std(peak_intervals)
    else:
        heart_rate = 0
        hrv = 0

    features['heart_rate'] = heart_rate
    features['hrv'] = hrv

    # Estadísticas básicas
    features['ppg_mean'] = np.mean(segment)
    features['ppg_std'] = np.std(segment)
    features['ppg_max'] = np.max(segment)
    features['ppg_min'] = np.min(segment)

    # Entropía espectral
    fft_vals = fft(segment)
    fft_power = np.abs(fft_vals)
    fft_power_norm = fft_power / np.sum(fft_power)
    features['ppg_entropy'] = entropy(fft_power_norm)

    return features

def extract_features_from_segments_ecg(segments, fs):
    """
    Extrae características de una lista de segmentos ECG.

    Args:
        segments (list): Lista de segmentos ECG.
        fs (int): Frecuencia de muestreo.

    Returns:
        list: Lista de diccionarios de características.
    """
    features_list = []
    for segment in segments:
        features = extract_features_ecg(segment, fs)
        features_list.append(features)
    return features_list

def extract_features_from_segments_ppg(segments, fs):
    """
    Extrae características de una lista de segmentos PPG.

    Args:
        segments (list): Lista de segmentos PPG.
        fs (int): Frecuencia de muestreo.

    Returns:
        list: Lista de diccionarios de características.
    """
    features_list = []
    for segment in segments:
        features = extract_features_ppg(segment, fs)
        features_list.append(features)
    return features_list
