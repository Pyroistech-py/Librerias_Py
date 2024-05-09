# -*- coding: utf-8 -*-
"""
Created on Wed May  8 08:21:39 2024

@author: amaia
"""

from sqlalchemy import create_engine, inspect
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import json
np.set_printoptions(threshold=np.inf)
from tqdm import tqdm


class Lector_SQL:
    @staticmethod   
    def leer_sql_datos(path, tabla, configuracion, consulta = None, serializado = False, unido= True):
    
        engine = create_engine('sqlite:///' + path)
        
        #WL
        consulta_wl = """
            SELECT WL
            FROM CONFIGURACION
            WHERE configuracion = :configuracion
        """
        if not configuracion:
            wl = None
        else:
            wl_sql = pd.read_sql_query(consulta_wl, con=engine, params={'configuracion': configuracion}).iloc[0, 0]
            wl = np.array([float(num) for num in wl_sql.strip("[]").split()])
        
        #DATOS
        df = pd.read_sql_table(tabla, con=engine) if consulta is None else pd.read_sql_query(consulta, con=engine)
        
        if 'DATOS' in tabla:
            if not serializado:
                data_arrays = np.array([np.array(json.loads(vec)) for vec in tqdm(df['DATOS'], desc='Procesando datos')])  
                df_datos = pd.DataFrame(data_arrays, columns=wl)
                df_info = df.drop('DATOS', axis=1)
    
                return pd.concat([df_info, df_datos], axis=1) if unido else (df_datos, df_info)
        else:
            df_info = df.drop('DATOS', axis=1)
            df_datos = df['DATOS']
            return df if unido else (df_datos, df_info)
        
    @staticmethod    
    def leer_sql_info(path, tabla):
        engine = create_engine('sqlite:///' + path)
        return pd.read_sql_table(tabla, con=engine) 
    
    @staticmethod
    def tablas_sql(path):
        # Crear el engine
        engine = create_engine('sqlite:///' + path)  # Asegúrate de proporcionar el path correcto al archivo .db
        
        # Usar la función inspect para obtener los nombres de las tablas
        inspector = inspect(engine)
        tablas = inspector.get_table_names()
        
        print("Tablas en la base de datos:")
        for tabla in tablas:
            print("\n#############################\n", tabla, "\n#############################\n")
        
            # Obtener detalles de las columnas para cada tabla
            print(f"Esquema de la tabla {tabla}:")
            columnas = inspector.get_columns(tabla)
            i=0
            for col in columnas:
                print(f"Col {i}: {col['name']}, Tipo: {col['type']}")
                i+=1
                
    @staticmethod
    def esquema_tabla(path, tabla):
        # Crear el engine
        engine = create_engine('sqlite:///' + path)  # Asegúrate de proporcionar el path correcto al archivo .db
        
        # Usar la función inspect para obtener los nombres de las tablas
        inspector = inspect(engine)
            
        print("\n#############################\n", tabla, "\n#############################\n")
    
        # Obtener detalles de las columnas para cada tabla
        print(f"Esquema de la tabla {tabla}:")
        columnas = inspector.get_columns(tabla)
        i= 0
        for col in columnas:
            print(f"Col {i}: {col['name']}, Tipo: {col['type']}")
            i+=1


# #Ejemplos de uso

# from Lector_SQL import Lector_SQL as lector

# path = 'path del archivo .db'
# tabla = 'DATOS_DISOLUCIONES'
# configuracion = 'CONFIG_0'

# #Consulta = manera de filtrar datos en sql (tb se pueden cargar las tablas completas y filtrar el df en python)
# #Ejemplo de consulta: filtrado por adulterante
# adulterante = 'GIRASOL' #En mayusculas
# consulta = f"""
#     SELECT d.*
#     FROM {tabla} d
#     INNER JOIN INFO i ON d.PY_ID = i.PY_ID
#     WHERE UPPER(i.ADULTERANTE) LIKE '%{adulterante}%'
# """

# #Ejemplo de consulta: filtrado por adulterante y concentracion de estigmastadienos
# estigs_value=0.2
# consulta2=f"""
#     SELECT d.*
#     FROM {tabla} d
#     INNER JOIN INFO i ON d.PY_ID = i.PY_ID
#     WHERE UPPER(i.ADULTERANTE) LIKE '%{adulterante}%'
#     AND i.ESTIGS > {estigs_value}
# """

# lector.esquema_tabla(path, 'INFO')
# lector.tablas_sql(path)
# df_base_info = lector.leer_sql_info(path, 'INFO')
# df = lector.leer_sql_datos(path, tabla, configuracion)
# df = lector.leer_sql_datos(path, tabla, configuracion, consulta)

