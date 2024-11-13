# signal_segmentation.py

def segment_signal(signal, window_size):
    """
    Segmenta una señal en ventanas de tamaño especificado.

    Args:
        signal (array): Señal de entrada.
        window_size (int): Tamaño de la ventana en muestras.

    Returns:
        list: Lista de segmentos de señal.
    """
    segments = []
    total_length = len(signal)
    for start in range(0, total_length - window_size + 1, window_size):
        segment = signal[start:start + window_size]
        segments.append(segment)
    return segments

def segment_signals(signals, window_size):
    """
    Segmenta una lista de señales.

    Args:
        signals (list): Lista de señales.
        window_size (int): Tamaño de la ventana en muestras.

    Returns:
        list: Lista de listas de segmentos.
    """
    all_segments = []
    for signal in signals:
        segments = segment_signal(signal, window_size)
        all_segments.extend(segments)
    return all_segments

def segment_ecg_and_ppg(ecg_signals, ppg_signals, window_size_ecg, window_size_ppg):
    """
    Segmenta las señales ECG y PPG.

    Args:
        ecg_signals (list): Lista de señales ECG.
        ppg_signals (list): Lista de señales PPG.
        window_size_ecg (int): Tamaño de ventana para ECG.
        window_size_ppg (int): Tamaño de ventana para PPG.

    Returns:
        tuple: Segmentos de ECG y PPG.
    """
    segments_ecg = segment_signals(ecg_signals, window_size_ecg)
    segments_ppg = segment_signals(ppg_signals, window_size_ppg)

    return segments_ecg, segments_ppg
