# data_processing.py

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

def load_data():
    """
    Carga los datos de ECG, PPG y las etiquetas desde archivos CSV.
    """
    # Reemplaza las rutas con las de tus archivos
    ecg_data = pd.read_csv('data/ecg_data.csv')
    ppg_data = pd.read_csv('data/ppg_data.csv')
    labels = pd.read_csv('data/labels.csv')
    
    return ecg_data, ppg_data, labels

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs  # Frecuencia de Nyquist
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_filter(data, lowcut, highcut, fs, order=4):
    """
    Aplica un filtro pasa banda Butterworth a las señales.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def preprocess_signals(ecg_data, ppg_data):
    """
    Preprocesa las señales de ECG y PPG: filtrado y normalización.
    """
    fs = 250  # Frecuencia de muestreo (Hz)
    lowcut = 0.5
    highcut = 40.0
    
    # Filtrar las señales
    ecg_filtered = ecg_data.apply(lambda x: apply_filter(x, lowcut, highcut, fs), axis=1)
    ppg_filtered = ppg_data.apply(lambda x: apply_filter(x, lowcut, highcut, fs), axis=1)
    
    # Normalización Z-score
    scaler_ecg = StandardScaler()
    ecg_normalized = scaler_ecg.fit_transform(ecg_filtered)
    
    scaler_ppg = StandardScaler()
    ppg_normalized = scaler_ppg.fit_transform(ppg_filtered)
    
    return ecg_normalized, ppg_normalized
