# generate_and_save_data.py

from synthetic_data_generator import generate_synthetic_ecg, generate_synthetic_ppg, save_signal_to_csv, plot_signal

def main():
    # Parámetros comunes
    fs_ecg = 360  # Frecuencia de muestreo ECG
    fs_ppg = 125  # Frecuencia de muestreo PPG
    duration = 10  # Duración en segundos
    heart_rate_normal = 70  # Frecuencia cardíaca normal
    heart_rate_afib = 110   # Frecuencia cardíaca en AFib (suele ser más alta)

    # Generar señales ECG normales
    t_ecg_normal, ecg_normal = generate_synthetic_ecg(fs_ecg, duration, heart_rate_normal, afib=False)
    save_signal_to_csv(t_ecg_normal, ecg_normal, 'ecg_normal.csv')
    plot_signal(t_ecg_normal, ecg_normal, 'ECG Normal')

    # Generar señales ECG con AFib
    t_ecg_afib, ecg_afib = generate_synthetic_ecg(fs_ecg, duration, heart_rate_afib, afib=True)
    save_signal_to_csv(t_ecg_afib, ecg_afib, 'ecg_afib.csv')
    plot_signal(t_ecg_afib, ecg_afib, 'ECG con Fibrilación Auricular')

    # Generar señales PPG normales
    t_ppg_normal, ppg_normal = generate_synthetic_ppg(fs_ppg, duration, heart_rate_normal, afib=False)
    save_signal_to_csv(t_ppg_normal, ppg_normal, 'ppg_normal.csv')
    plot_signal(t_ppg_normal, ppg_normal, 'PPG Normal')

    # Generar señales PPG con AFib
    t_ppg_afib, ppg_afib = generate_synthetic_ppg(fs_ppg, duration, heart_rate_afib, afib=True)
    save_signal_to_csv(t_ppg_afib, ppg_afib, 'ppg_afib.csv')
    plot_signal(t_ppg_afib, ppg_afib, 'PPG con Fibrilación Auricular')

if __name__ == "__main__":
    main()
