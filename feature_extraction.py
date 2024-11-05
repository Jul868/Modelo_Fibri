# feature_extraction.py

import numpy as np
import pandas as pd

def extract_ecg_features(ecg_signals):
    """
    Extrae características del ECG.
    """
    features = []
    for signal in ecg_signals:
        feature_dict = {}
        feature_dict['ecg_mean'] = np.mean(signal)
        feature_dict['ecg_std'] = np.std(signal)
        feature_dict['ecg_max'] = np.max(signal)
        feature_dict['ecg_min'] = np.min(signal)
        # Agrega más características si es necesario
        features.append(feature_dict)
    return pd.DataFrame(features)

def extract_ppg_features(ppg_signals):
    """
    Extrae características del PPG.
    """
    features = []
    for signal in ppg_signals:
        feature_dict = {}
        feature_dict['ppg_mean'] = np.mean(signal)
        feature_dict['ppg_std'] = np.std(signal)
        feature_dict['ppg_max'] = np.max(signal)
        feature_dict['ppg_min'] = np.min(signal)
        # Agrega más características si es necesario
        features.append(feature_dict)
    return pd.DataFrame(features)

def extract_features(ecg_processed, ppg_processed):
    """
    Combina las características de ECG y PPG.
    """
    ecg_features = extract_ecg_features(ecg_processed)
    ppg_features = extract_ppg_features(ppg_processed)
    features = pd.concat([ecg_features, ppg_features], axis=1)
    return features
