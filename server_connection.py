# -*- coding: utf-8 -*-

from ftplib import FTP_TLS
import io
import pandas as pd
import time


class FTPDownloader:
    def __init__(self, servidor_ftp="", usuario="", contraseña=""):
        self.servidor_ftp = servidor_ftp
        self.usuario = usuario
        self.contraseña = contraseña

    def conectar_ftps(self):
        ftps = FTP_TLS(self.servidor_ftp)
        ftps.login(user=self.usuario, passwd=self.contraseña)
        ftps.prot_p()
        return ftps

    def listar_directorios(self, ruta_carpeta="/"):
        try:
            with self.conectar_ftps() as ftps:
                ftps.cwd(ruta_carpeta)
                archivos = ftps.nlst()
                print(f"Archivos y  directorios en {ruta_carpeta}: {archivos}")
                return archivos
        except Exception as e:
            print(f"Error al listar los directorios: {e}")

    def descargar_y_cargar_csv(self, nombre_archivo, ruta_carpeta):
        try:
            with self.conectar_ftps() as ftps:
                ftps.cwd(ruta_carpeta)

                print(
                    f"Conexión exitosa al directorio {ruta_carpeta}. Leyendo archivo {nombre_archivo}..."
                )

                tiempo_inicio = time.time()

                buffer = io.BytesIO()
                ftps.retrbinary(f"RETR {nombre_archivo}", buffer.write)
                buffer.seek(0)

                df = pd.read_csv(buffer)

                tiempo_final = time.time()

                print(
                    f"Tiempo total de la operación: {tiempo_final - tiempo_inicio} segundos."
                )

                return df

        except Exception as e:
            print(
                f"Error al conectar o al trabajar con el FTPS o al leer el archivo: {e}"
            )


# =============================================================================
# Example 
# =============================================================================


# servidor_ftp="your_server_IP_adress"
# usuario="user"
# contraseña="password"


# # Definir el descargador
# downloader = FTPDownloader(servidor_ftp, usuario, contraseña)

# # Listar directorios y ficheros
# downloader.listar_directorios()

# #utilizar_lo_anterior_para_ir_definiendo_la_ruta_a_la_carpeta
# nombre_archivo = "your_file.txt",
# ruta_carpeta = "path/to/the/file",

# df = downloader.descargar_y_cargar_csv(
#     nombre_archivo = nombre_archivo,
#     ruta_carpeta = ruta_carpeta,
# )
