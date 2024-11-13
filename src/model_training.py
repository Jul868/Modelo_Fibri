# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_dataset(features_normal, features_afib):
    """
    Crea un DataFrame combinando características de ritmos normales y de fibrilación auricular.

    Args:
        features_normal (list): Lista de diccionarios con características de ritmos normales.
        features_afib (list): Lista de diccionarios con características de fibrilación auricular.

    Returns:
        DataFrame: DataFrame con características y etiquetas.
    """
    # Crear DataFrames a partir de las listas de características
    df_normal = pd.DataFrame(features_normal)
    df_normal['label'] = 0  # Etiqueta para ritmo normal

    df_afib = pd.DataFrame(features_afib)
    df_afib['label'] = 1  # Etiqueta para fibrilación auricular

    # Combinar los DataFrames
    df = pd.concat([df_normal, df_afib], ignore_index=True)
    return df

def prepare_data(df):
    """
    Prepara los datos para el entrenamiento del modelo.

    Args:
        df (DataFrame): DataFrame con características y etiquetas.

    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler
    """
    # Separar características y etiquetas
    X = df.drop('label', axis=1)
    y = df['label']

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalamiento de características
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

def train_model(X_train, y_train, input_shape):
    """
    Entrena un modelo de red neuronal en Keras.

    Args:
        X_train (array): Características de entrenamiento.
        y_train (array): Etiquetas de entrenamiento.
        input_shape (int): Dimensión de entrada del modelo.

    Returns:
        keras.Model: Modelo entrenado.
    """
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    # Guardar en formato SavedModel (TensorFlow)
    model.save('models/modelo_afib.h5')
    print("Modelo guardado como 'modelo_afib.h5'.")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    print("Modelo convertido a formato TensorFlow Lite.")
    with open('modelo_afib.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Modelo guardado como 'modelo_afib.tflite'.")


    return model
