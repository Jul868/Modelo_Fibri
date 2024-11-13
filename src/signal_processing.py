# signal_processing.py

from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Aplica un filtro pasa banda a la señal.

    Args:
        data (array): Señal de entrada.
        lowcut (float): Frecuencia de corte baja.
        highcut (float): Frecuencia de corte alta.
        fs (int): Frecuencia de muestreo de la señal.
        order (int): Orden del filtro.

    Returns:
        array: Señal filtrada.
    """
    nyq = 0.5 * fs  # Frecuencia de Nyquist
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def filter_signals(signals, lowcut, highcut, fs):
    """
    Aplica el filtro a una lista de señales.

    Args:
        signals (list): Lista de señales a filtrar.
        lowcut (float): Frecuencia de corte baja.
        highcut (float): Frecuencia de corte alta.
        fs (int): Frecuencia de muestreo.

    Returns:
        list: Lista de señales filtradas.
    """
    filtered_signals = []
    for signal in signals:
        filtered_signal = bandpass_filter(signal[:, 0], lowcut, highcut, fs)
        filtered_signals.append(filtered_signal)
    return filtered_signals

def filter_ecg_and_ppg(ecg_signals, ppg_signals, fs_ecg, fs_ppg):
    """
    Filtra las señales ECG y PPG.

    Args:
        ecg_signals (list): Lista de señales ECG.
        ppg_signals (list): Lista de señales PPG.
        fs_ecg (int): Frecuencia de muestreo ECG.
        fs_ppg (int): Frecuencia de muestreo PPG.

    Returns:
        tuple: Señales ECG y PPG filtradas.
    """
    # Parámetros de filtrado para ECG
    ecg_lowcut = 0.5
    ecg_highcut = 45.0

    # Parámetros de filtrado para PPG
    ppg_lowcut = 0.5
    ppg_highcut = 8.0  # Las frecuencias de interés en PPG suelen ser más bajas

    filtered_ecg = filter_signals(ecg_signals, ecg_lowcut, ecg_highcut, fs_ecg)
    filtered_ppg = filter_signals(ppg_signals, ppg_lowcut, ppg_highcut, fs_ppg)

    return filtered_ecg, filtered_ppg
