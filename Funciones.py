# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 08:45:42 2023

@author: amaia
"""

import numpy as np
import pandas as pd
import unicodedata

from scipy.signal import savgol_filter
from scipy.signal import detrend

class funciones:
    
    #Funcion para encontrar la columna mas cercana a la introducida
    def encontrar_cercano(self, a, a0):
        idx = np.abs(a - a0).argmin()
        idx = a[idx]
        return np.where(a == idx)[0][0], idx
   
    def aplicar_detrend(self, df):
        # Inicializar df_detrended con las mismas columnas que df
        df_detrended = pd.DataFrame(columns=df.columns)
        for index, row in df.iterrows():
            # Detrend la fila y asignarla al DataFrame df_detrended
            df_detrended.loc[index] = detrend(row, axis =0)
        return df_detrended
    
    def calcular_absorbancia(self, df, ref):
        df[(df <= 0) | df.isna()] = 1e-10
        ref[(ref <= 0) | ref.isna()] = 1e-10
        absorbancia = -np.log10(df / np.array(ref)) 
        return absorbancia
    
    def snv(self, input_data):
        # Restar la media y dividir por la desviación estándar de cada fila
        mean = input_data.mean(axis=1) 
        std = input_data.std(axis=1)
        snv = (input_data.sub(mean, axis=0)).div(std, axis=0)
        return snv
    
    def quitar_tildes(self, input_str):
        # Normalizar la cadena de texto
        nfkd_form = unicodedata.normalize('NFD', input_str)
        # Filtrar solo los caracteres alfabéticos básicos
        only_ascii = nfkd_form.encode('ASCII', 'ignore')
        return only_ascii.decode('ASCII')

    def filtro_sg(self, df, w, p, d):
        sg = savgol_filter(df,w, p, deriv=d)
        df_sg=pd.DataFrame(sg, index=df.index, columns=df.columns)
        return df_sg