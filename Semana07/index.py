# Importar bibliotecas necesarias
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paso 1: Cargar la imagen
# Reemplaza 'imagen_kotosh.jpg' con el nombre de tu imagen
imagen = cv2.imread('logo-Cetpro.png')

# Paso 2: Convertir la imagen a escala de grises
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Paso 3: Detectar bordes con Canny (detección básica de contornos)
bordes = cv2.Canny(imagen_gris, 100, 200)

# Paso 4: Encontrar contornos
contornos, jerarquia = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Paso 5: Dibujar los contornos en la imagen original
imagen_con_contornos = imagen.copy()
cv2.drawContours(imagen_con_contornos, contornos, -1, (0, 255, 0), 2)

# Mostrar los resultados
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Imagen Original")
plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Contornos Detectados")
plt.imshow(cv2.cvtColor(imagen_con_contornos, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
