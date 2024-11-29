# Archivo para procesar los datos de los pacientes
import os
import pandas as pd
import numpy as np
import math

# Función para calcular la distancia
def distance(x, y, z):
    return math.sqrt(x**2 + y**2 + z**2)

# Función para procesar un archivo CSV y agregar las columnas de tiempo, distancia y suma máxima
def preprocess_file(filepath, output_dir):
    # Detectar el ventaneo desde el nombre del archivo (e.g., v07)
    filename = os.path.basename(filepath)
    ventaneo = int(filename.split("_v")[1][:2])  # Extrae el valor numérico después de "_v"

    # Cargar datos y calcular la columna de tiempo
    data = pd.read_csv(filepath)
    data["time"] = data.index * ventaneo  # Calcula el tiempo según el ventaneo

    # Calcular la columna de distancia
    if {"Des_x", "Des_y", "Des_z"}.issubset(data.columns):  # Verificar que las columnas existen
        data["Distancia"] = data.apply(lambda row: distance(row["Des_x"], row["Des_y"], row["Des_z"]), axis=1)

    # Reorganizar columnas para que "time" sea la primera
    columns = ["time"] + [col for col in data.columns if col != "time"]
    data = data[columns]

    # Crear columnas de Intervalos de ciertas columnas
    # Calcular la columna de distancia
    if {"Veloc", "Acele","Des_x", "Des_y", "Des_z","Presn","VelPr","AcePr","Distancia"}.issubset(data.columns):  # Verificar que las columnas existen
        n = 10
        columns_to_graph = ['Veloc', 'Acele', 'Des_x', 'Des_y', 'Des_z', 'Presn', 'VelPr', 'AcePr', 'Distancia']

        intervals_cols = []
        for column in columns_to_graph:
            min_val = data[column].min()
            max_val = data[column].max()
            intervalos = np.linspace(min_val, max_val, n+1)    

            #Usa pd.cut con 'labels' como el rango de intervalos en formato de string
            data['Intervalo_'+column] = pd.cut(
                data[column],
                bins = intervalos,
                labels = [f'[{intervalos[i]:.2f}, {intervalos[i+1]:.2f})' for i in range(n)],
                include_lowest=True
            )

            intervals_cols.append('Intervalo_'+column)

    # Crear directorio de salida si no existe
    patient_folder = filepath.split(os.sep)[-3]  # Carpeta del paciente (e.g., P07)
    session_folder = filepath.split(os.sep)[-2]  # Subcarpeta de la sesión (e.g., 01ene02)
    output_path = os.path.join(output_dir, patient_folder, session_folder)

    os.makedirs(output_path, exist_ok=True)
    output_filepath = os.path.join(output_path, filename)
    
    # Guardar archivo procesado
    data.to_csv(output_filepath, index=False)

# Función para recorrer todas las carpetas y procesar los archivos
def preprocess_all_data(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".csv"):  # Procesar solo archivos CSV
                filepath = os.path.join(root, file)
                preprocess_file(filepath, output_dir)

# Directorios de entrada y salida
input_directory = "C:/Users/betox/Documents/Academico/7moSemestre/EntregaFinal/Dashboard/archivos"  # Carpeta con los datos originales (carpeta 'archivos')
output_directory = "C:/Users/betox/Documents/Academico/7moSemestre/EntregaFinal/Dashboard/data"  # Carpeta para guardar los datos procesados (carpeta 'data')

# Ejecutar preprocesamiento
preprocess_all_data(input_directory, output_directory)
print("Preprocesamiento completado. Los archivos procesados se guardaron en la carpeta:", output_directory)