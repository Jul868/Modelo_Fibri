# main.py

from data_processing import load_data, preprocess_signals
from feature_extraction import extract_features
from model import train_model, evaluate_model, save_model

def main():
    # Cargar y preprocesar los datos
    ecg_data, ppg_data, labels = load_data()
    ecg_processed, ppg_processed = preprocess_signals(ecg_data, ppg_data)
    
    # Extraer características
    features = extract_features(ecg_processed, ppg_processed)
    
    # Entrenar y evaluar el modelo
    model, X_test, y_test = train_model(features, labels)
    evaluate_model(model, X_test, y_test)
    
    # Guardar el modelo entrenado
    save_model(model, 'trained_model.pkl')

if __name__ == '__main__':
    main()
