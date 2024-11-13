# data_loader.py

import pandas as pd
import glob

def load_csv_signal(file_path):
    """
    Carga una señal desde un archivo CSV.

    Args:
        file_path (str): Ruta al archivo CSV.

    Returns:
        array: Señal cargada.
    """
    df = pd.read_csv(file_path)
    print(f"Archivo {file_path} cargado, primeras filas:")
    print(df.head())
    signal = df['Signal'].values
    return signal

def get_file_paths(data_directory, extension='*.csv'):
    """
    Obtiene todas las rutas de archivos en un directorio dado.

    Args:
        data_directory (str): Ruta al directorio de datos.
        extension (str): Extensión de los archivos a buscar.

    Returns:
        list: Lista de rutas de archivos.
    """
    file_paths = glob.glob(f"{data_directory}/{extension}")
    return file_paths
