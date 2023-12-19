# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 08:45:42 2023

@author: amaia
"""

import numpy as np
import pandas as pd
import unicodedata

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.signal import savgol_filter
from scipy.signal import detrend

import traceback
import re


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

    def plot_graphs(self,
                df,
                info_df, 
                columna_datos,
                grid_size=(1, 1), 
                position=(0, 0), 
                fig=None, 
                figsize=(12, 8), 
                x_axis_ticks=None,
                cmap_selected=plt.cm.tab10, 
                title=None,
                x_label=None,
                y_label=None,
                active_grid=True,
                show_colorbar=False,  # Colorbar no se usará para datos tipo str
                show_legend=True,
                legend_loc="best",
                title_legend=None
                ):
        
            try:
                if fig is None:
                    fig = plt.figure(figsize=figsize)
                ax = plt.subplot2grid(grid_size, position, fig=fig)
        
                if x_axis_ticks is None:
                    x_axis_ticks = df.columns
        
                # Crear un mapeo de colores para valores únicos de tipo str
                
                info_df[columna_datos] = info_df[columna_datos].astype(str)
    
                
                unique_str_values = sorted(info_df[columna_datos].unique(), key=lambda x: self.natural_keys(x))
                color_map = {value: cmap_selected(i) for i, value in enumerate(unique_str_values)}
        
                for index, row in df.iterrows():
                    str_value = info_df.loc[index, columna_datos]
                    color = color_map.get(str_value, "black")  # Color por defecto si no se encuentra en el mapeo
        
                    ax.plot(x_axis_ticks, row, color=color)
        
                ax.set_title(title)
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.grid(active_grid)
                
                if show_legend:
                    handles = [mpatches.Patch(color=color_map[value], label=value) for value in unique_str_values]
                    ax.legend(handles=handles, title=title_legend, loc=legend_loc)
        
                return fig
        
            except Exception as e:
                # Aquí puedes manejar el error como prefieras
                print(f"Ha ocurrido un error: {e}")
                traceback.print_exc()
                # O devolver un mensaje de error
                # return f"Error en la graficación: {e}"
    
            return fig
    
    def atoi(self, text):
            return int(text) if text.isdigit() else text
        
    def natural_keys(self, text):
            '''
            Algoritmo para ordenar strings que contienen números de manera natural
            '''
            if isinstance(text, bytes):
                text = text.decode('utf-8')  # Decodificar de bytes a str
    
            return [self.atoi(c) for c in re.split(r'(\d+)', text)]

'''
uso de plot_graph
fig = plot_graphs(df, info_df, columna, grid_size=(2,2), position=(0,1), fig=None, show_colorbar=False, show_legend=True)
fig = plot_graphs(df, info_df, columna2, grid_size=(2,2), position=(1, 0), fig=fig, show_colorbar=False, show_legend=True)
plt.tight_layout()
plt.show()
'''
