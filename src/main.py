# main.py

from data_loader import load_csv_signal
from signal_processing import filter_signals
from signal_segmentation import segment_signal
from feature_extraction import extract_features_ecg, extract_features_ppg
from model_training import create_dataset, prepare_data, train_model
from model_evaluation import evaluate_model

def main():
    # Rutas a los archivos CSV
    ecg_normal_file = 'data/ecg_normal.csv'
    ecg_afib_file = 'data/ecg_afib.csv'
    ppg_normal_file = 'data/ppg_normal.csv'
    ppg_afib_file = 'data/ppg_afib.csv'

    # Cargar señales desde archivos CSV
    ecg_normal_signal = load_csv_signal(ecg_normal_file)
    ecg_afib_signal = load_csv_signal(ecg_afib_file)
    ppg_normal_signal = load_csv_signal(ppg_normal_file)
    ppg_afib_signal = load_csv_signal(ppg_afib_file)

    # Verificar la longitud de las señales
    print(f"ECG Normal Signal Length: {len(ecg_normal_signal)} samples")
    print(f"ECG AFib Signal Length: {len(ecg_afib_signal)} samples")
    print(f"PPG Normal Signal Length: {len(ppg_normal_signal)} samples")
    print(f"PPG AFib Signal Length: {len(ppg_afib_signal)} samples")

    # Parámetros
    fs_ecg = 360  # Frecuencia de muestreo ECG
    fs_ppg = 125  # Frecuencia de muestreo PPG
    window_duration = 2  # Duración de la ventana en segundos
    window_size_ecg = int(fs_ecg * window_duration)
    window_size_ppg = int(fs_ppg * window_duration)

    # Segmentación
    segments_ecg_normal = segment_signal(ecg_normal_signal, window_size_ecg)
    segments_ecg_afib = segment_signal(ecg_afib_signal, window_size_ecg)
    segments_ppg_normal = segment_signal(ppg_normal_signal, window_size_ppg)
    segments_ppg_afib = segment_signal(ppg_afib_signal, window_size_ppg)

    # Verificar la cantidad de segmentos
    print(f"Segments ECG Normal: {len(segments_ecg_normal)}")
    print(f"Segments ECG AFib: {len(segments_ecg_afib)}")
    print(f"Segments PPG Normal: {len(segments_ppg_normal)}")
    print(f"Segments PPG AFib: {len(segments_ppg_afib)}")

    # Extracción de características
    features_ecg_normal = [extract_features_ecg(segment, fs_ecg) for segment in segments_ecg_normal]
    features_ppg_normal = [extract_features_ppg(segment, fs_ppg) for segment in segments_ppg_normal]

    features_ecg_afib = [extract_features_ecg(segment, fs_ecg) for segment in segments_ecg_afib]
    features_ppg_afib = [extract_features_ppg(segment, fs_ppg) for segment in segments_ppg_afib]

    # Combinar características de ECG y PPG
    features_normal = []
    for ecg_feat, ppg_feat in zip(features_ecg_normal, features_ppg_normal):
        combined_features = {**ecg_feat, **ppg_feat}
        features_normal.append(combined_features)

    features_afib = []
    for ecg_feat, ppg_feat in zip(features_ecg_afib, features_ppg_afib):
        combined_features = {**ecg_feat, **ppg_feat}
        features_afib.append(combined_features)

    # Verificar la cantidad de características
    print(f"Features Normal Length: {len(features_normal)}")
    print(f"Features AFib Length: {len(features_afib)}")

    # Creación del conjunto de datos
    df = create_dataset(features_normal, features_afib)

    # Verificar el DataFrame
    print("DataFrame df:")
    print(df.head())
    print(f"Total samples: {len(df)}")

    # Preparación de datos
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    # Obtener la dimensión de entrada
    input_shape = X_train.shape[1]

    # Entrenamiento del modelo
    model = train_model(X_train, y_train, input_shape)

    # Evaluación del modelo
    results = evaluate_model(model, X_test, y_test)

# main.py

# ... [código anterior] ...

    # Mostrar resultados
    print(f"Precisión del modelo: {results['accuracy']:.2f}%")
    print("Matriz de confusión:")
    print(results['confusion_matrix'])
    print(results['classification_report'])



if __name__ == "__main__":
    main()
