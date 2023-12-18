# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:26:44 2023

@author: amaia
"""
import numpy as np
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np

class Lector_BBDD:
    def __init__(self, path_bd, wl_column, data_column, selected_columns=None):
        # Constructor de la clase. Inicializa los atributos de la instancia con los valores proporcionados.
        # path_bd: Ruta al archivo de base de datos.
        # wl_column: Nombre de la columna que contiene datos de longitud de onda.
        # data_column: Nombre de la columna que contiene los datos a ser procesados.
        # selected_columns: Lista opcional de columnas adicionales a seleccionar.
        self.path_bd = path_bd
        self.wl_column = wl_column
        self.data_column = data_column
        self.selected_columns = selected_columns
        
        self.df_bbdd, self.df_info = self.leer_datos()
        
    def leer_datos(self):
        # Método para leer y procesar los datos de la base de datos.
        
        # Lee el archivo CSV usando pandas y separa los campos por tabulaciones.
        df_bd = pd.read_csv(self.path_bd,  sep='\t')    
        
        # Procesa la columna de datos para convertir los valores de cadena en listas de números flotantes.
        self.data = []
        for index, row in df_bd.iterrows():
            try:
                row_data = [float(valor) for valor in str(row[self.data_column]).strip("[]").split(',')]
            except ValueError:
                row_data = [(valor) for valor in row[self.data_column].strip("[]").split(',')]        
            self.data.append(row_data)
    
    
        # Extrae y procesa los valores de longitud de onda de la primera fila, convirtiéndolos en flotantes.
        wl = [float(num.replace(',', '')) for num in df_bd.at[0, self.wl_column].strip("[]").split() ]
        
        # Crea un DataFrame de pandas con los datos procesados, utilizando las longitudes de onda como nombres de columnas.
        df_bbdd= pd.DataFrame(self.data, columns=wl)
        
        # Inicializa un DataFrame vacío para información adicional.
        df_info=pd.DataFrame()
        
        # Si se han especificado columnas seleccionadas, se añaden al DataFrame df_info.
        if self.selected_columns is not None:
            if all(column in df_bd.columns for column in self.selected_columns):
                for column in self.selected_columns:
                    df_info[column] = df_bd[column]
               
        # Devuelve dos DataFrames: uno con los datos procesados y otro con información adicional si está disponible.
        return df_bbdd, df_info
    
    def obtener_nombres_columnas(self):
        # Método para obtener los nombres de las columnas del archivo de base de datos.

        # Lee solamente los encabezados del archivo.
        df_temp = pd.read_csv(self.path_bd, sep='\t', nrows=0)

        # Devuelve la lista de nombres de columnas.
        return df_temp.columns.tolist()


"""
EJEMPLO DE USO DE LA LIBRERIA

* Descomentar para probar
"""
# #########

# from Lector_BBDD_AGROFLUID import Lector_BBDD

# # Carga de datos
# lector = Lector_BBDD('DATOS/BBDD_AGROFLUID_005.txt', 'WL_VIS', 'DATA_VIS', ['ACEITE', 'CONCENTRADO'])
# df = lector.df_bbdd

# # Obtener las columnas que hay en el dataset
# nombres_columnas = lector.obtener_nombres_columnas()

# # Preparar los arrays con los datos
# wl = df.columns.values
# datos = df.values.astype('float32')

# aceite = lector.df_info['ACEITE'].values
# concentrado = lector.df_info['CONCENTRADO'].values
