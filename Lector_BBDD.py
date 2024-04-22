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
import pyarrow.parquet as pq
from sqlalchemy import create_engine

class Lector_BBDD:
    # Constructor de la clase. Inicializa los atributos de la instancia con los valores proporcionados.
    # path_bd: Ruta al archivo de base de datos.
    # wl_column: Nombre de la columna que contiene datos de longitud de onda.
    # data_column: Nombre de la columna que contiene los datos a ser procesados.
    # selected_columns: Lista opcional de columnas adicionales a seleccionar.
        
        
    def leer_datos(self, path_bd, wl_column, data_column, selected_columns=None):
        # Método para leer y procesar los datos de la base de datos.
        self.path_bd = path_bd
        self.wl_column = wl_column
        self.data_column = data_column
        self.selected_columns = selected_columns
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
    
    def obtener_nombres_columnas(self,path_bd):
        self.path_bd = path_bd
        # Método para obtener los nombres de las columnas del archivo de base de datos.

        # Lee solamente los encabezados del archivo.
        df_temp = pd.read_csv(self.path_bd, sep='\t', nrows=0)

        # Devuelve la lista de nombres de columnas.
        return df_temp.columns.tolist()
    

    #Lector para parquet

    def leer_datos_pq(self, path_bd, wl_column, data_column, selected_columns=None):
            # Leer las columnas seleccionadas si se especifican
            if selected_columns is not None:
                df_info = pd.read_parquet(path_bd, columns=selected_columns)
        
            # Leer la columna de longitudes de onda
            wl = pd.read_parquet(path_bd, columns=[wl_column]).iloc[0, 0]
        
            # Leer los datos
            data_series = pd.read_parquet(path_bd, columns=[data_column])[data_column]
        
            # Procesar cada elemento en data_series
            processed_data = []
            for item in data_series:
                if item is None:
                    # Opción 1: Reemplazar None con una lista de NaN
                    processed_data.append([np.nan] * len(wl))
                    # Opción 2: Omitir las filas con None (descomentar la siguiente línea y comentar la anterior)
                    # continue
                else:
                    processed_data.append(item if pd.api.types.is_list_like(item) else [item])
        
            # Convertir la lista procesada a un DataFrame
            data_df = pd.DataFrame(processed_data, columns=wl)
        
            return data_df, df_info

    
    def obtener_nombres_columnas_pq(self, path_bd):
        parquet_file = pq.ParquetFile(path_bd)
        column_names=parquet_file.metadata.schema.names
        return column_names

class Lector_SQL():
    @staticmethod
    def create_engine_db(path):
        """ Crea y devuelve el motor de base de datos """
        return create_engine('sqlite:///' + path)
    
    @staticmethod
    def load_wl(engine, configuracion):
        """ Carga la configuración de la base de datos """
        consulta = """
            SELECT WL
            FROM DB_CONFIGURACION
            WHERE configuracion = :configuracion
        """
        if not configuracion:
            return None
        else:
            wl_sql = pd.read_sql_query(consulta, con=engine, params={'configuracion': configuracion}).iloc[0, 0]
            return np.array([float(num) for num in wl_sql.strip("[]").split()])

    @staticmethod
    def bytes_to_floats_vectorized(byte_data):
        # Primero convertimos los datos de bytes a un array de numpy
        byte_array = np.frombuffer(byte_data, dtype=np.uint8)
        
        # Calculamos cuántos números de doble precisión contiene el array
        num_floats = byte_array.size // 8
        
        # Reinterpretamos el array de bytes como un array de doubles
        return byte_array.view(np.float64)[:num_floats]

    @staticmethod
    def deserializar(df, wl):
        """ Transforma los datos de la columna 'DATOS' """
        data_arrays = np.vstack(df['DATOS'].map(Lector_SQL.bytes_to_floats_vectorized).tolist())
        return pd.DataFrame(data_arrays, columns=wl)
    
    @staticmethod
    def leer_sql(path, tabla, configuracion=None, consulta=None, serializado=False, unido=False):
        engine = Lector_SQL.create_engine_db(path)
        df = pd.read_sql_table(tabla, con=engine) if consulta is None else pd.read_sql_query(consulta, con=engine)
        
        if 'DATOS' in tabla:
            if not serializado:
                wl = Lector_SQL.load_wl(engine, configuracion) if configuracion else None
                
                df_datos = Lector_SQL.deserializar(df, wl)
                df_info = df.drop('DATOS', axis=1)
                
                return pd.concat([df_info, df_datos], axis=1) if unido else (df_datos, df_info)
            else:
                df_info = df.drop('DATOS', axis=1)
                df_datos = df['DATOS']
                return df if unido else (df_datos, df_info)
        else:
            return df




"""
EJEMPLO DE USO DE LA LIBRERIA

* Descomentar para probar
"""
# #########
# from Lector_BBDD import Lector_BBDD

# # Carga de datos
# lector = Lector_BBDD(path_BBDD.txt', 'WL_VIS', 'DATA_VIS', ['ACEITE', 'CONCENTRADO'])
# df = lector.df_bbdd

# # Obtener las columnas que hay en el dataset
# nombres_columnas = lector.obtener_nombres_columnas()

# # Preparar los arrays con los datos
# wl = df.columns.values
# datos = df.values.astype('float32')

# aceite = lector.df_info['ACEITE'].values
# concentrado = lector.df_info['CONCENTRADO'].values
