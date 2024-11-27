# Importar bibliotecas necesarias
import os
import numpy as np
import cv2  # OpenCV para procesar imágenes
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Función para cargar y preprocesar imágenes
def cargar_imagenes(directorio, size=(64, 64)):
    datos = []
    etiquetas = []
    clases = os.listdir(directorio)

    for idx, clase in enumerate(clases):
        ruta_clase = os.path.join(directorio, clase)
        for archivo in os.listdir(ruta_clase):
            ruta_archivo = os.path.join(ruta_clase, archivo)
            # Leer la imagen en escala de grises y redimensionarla
            img = cv2.imread(ruta_archivo, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, size)
                datos.append(img_resized.flatten())  # Convertir en vector
                etiquetas.append(idx)  # Etiqueta numérica de la clase
    return np.array(datos), np.array(etiquetas), clases

# Ruta al dataset (estructura de carpetas)
ruta_dataset = "dataset"  # Asegúrate de tener imágenes organizadas en carpetas
# dataset/
# ├── opciones_ocupacionales/
# │   ├── img1.jpg
# │   ├── img2.jpg
# ├── programas_estudio/
# │   ├── img3.jpg
# │   ├── img4.jpg

# Cargar imágenes y etiquetas
X, y, clases = cargar_imagenes(ruta_dataset)
print(f"Clases detectadas: {clases}")
print(f"Número de imágenes cargadas: {len(X)}")

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo SVM
modelo_svm = SVC(kernel="linear", random_state=42)
modelo_svm.fit(X_train, y_train)
print("Modelo entrenado con éxito.")

# Evaluar el modelo
y_pred = modelo_svm.predict(X_test)
precision = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {precision * 100:.2f}%")
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=clases))

# Función para predecir una imagen nueva
def predecir_imagen(modelo, ruta_imagen, clases, size=(64, 64)):
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img_resized = cv2.resize(img, size)
        img_flat = img_resized.flatten().reshape(1, -1)
        prediccion = modelo.predict(img_flat)
        return clases[prediccion[0]]
    else:
        return "Imagen no válida"

# Probar con una nueva imagen
ruta_imagen_prueba = "dataset/opciones_ocupacionales/hairdresser.jpg"  # Cambia por una imagen válida
categoria = predecir_imagen(modelo_svm, ruta_imagen_prueba, clases)
print(f"La imagen pertenece a la categoría: {categoria}")

# Mostrar la imagen con la predicción
img_prueba = cv2.imread(ruta_imagen_prueba, cv2.IMREAD_GRAYSCALE)
if img_prueba is not None:
    plt.imshow(img_prueba, cmap="gray")
    plt.title(f"Predicción: {categoria}")
    plt.axis("off")
    plt.show()
