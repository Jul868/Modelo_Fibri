# synthetic_data_generator.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_synthetic_ecg(fs, duration, heart_rate, afib=False):
    """
    Genera una señal sintética de ECG.

    Args:
        fs (int): Frecuencia de muestreo en Hz.
        duration (float): Duración de la señal en segundos.
        heart_rate (float): Frecuencia cardíaca en latidos por minuto.
        afib (bool): Si es True, genera una señal con fibrilación auricular.

    Returns:
        array: Señal ECG sintética.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    if afib:
        # Variabilidad significativa en el intervalo RR
        rr_intervals = np.random.normal(60 / heart_rate, 0.2, size=len(t))
        phases = np.cumsum(2 * np.pi * rr_intervals * fs / fs)
        signal = np.sin(phases)
    else:
        # Señal ECG normal utilizando una función senoidal básica
        f = heart_rate / 60  # Frecuencia cardíaca en Hz
        signal = np.sin(2 * np.pi * f * t)
    return t, signal

def generate_synthetic_ppg(fs, duration, heart_rate, afib=False):
    """
    Genera una señal sintética de PPG.

    Args:
        fs (int): Frecuencia de muestreo en Hz.
        duration (float): Duración de la señal en segundos.
        heart_rate (float): Frecuencia cardíaca en latidos por minuto.
        afib (bool): Si es True, genera una señal con fibrilación auricular.

    Returns:
        array: Señal PPG sintética.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    if afib:
        # Variabilidad significativa en el intervalo RR
        rr_intervals = np.random.normal(60 / heart_rate, 0.2, size=len(t))
        phases = np.cumsum(2 * np.pi * rr_intervals * fs / fs)
        signal = np.abs(np.sin(phases))
    else:
        # Señal PPG normal utilizando una función senoidal básica
        f = heart_rate / 60  # Frecuencia cardíaca en Hz
        signal = np.abs(np.sin(2 * np.pi * f * t))
    return t, signal

def save_signal_to_csv(t, signal, filename):
    """
    Guarda la señal en un archivo CSV.

    Args:
        t (array): Vector de tiempo.
        signal (array): Señal a guardar.
        filename (str): Nombre del archivo CSV.
    """
    df = pd.DataFrame({'Time': t, 'Signal': signal})
    df.to_csv(filename, index=False)
    print(f"Señal guardada en {filename}")

def plot_signal(t, signal, title):
    """
    Grafica la señal.

    Args:
        t (array): Vector de tiempo.
        signal (array): Señal a graficar.
        title (str): Título del gráfico.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal)
    plt.title(title)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.grid(True)
    plt.show()
