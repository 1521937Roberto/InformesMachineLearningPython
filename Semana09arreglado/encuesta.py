import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Catálogos de opciones
opciones_ocupacionales = [
    "Computación",
    "Cocina",
    "Electricidad",
    "Peluquería básica"
]

programas_estudio = [
    "Panadería y pastelería",
    "Mecánica de motos y vehículos",
    "Apoyo administrativo"
]

# DataFrame para almacenar respuestas
data = pd.DataFrame(columns=["Estudiante", "Categoría", "Carrera"])

# Encuesta a 10 estudiantes
print("Bienvenidos a la Encuesta de Preferencias de Kotosh")
for estudiante in range(1, 11):
    print(f"\nEstudiante {estudiante}:")
    while True:
        try:
            respuesta = int(input("¿Qué estudia: (1) Opciones Ocupacionales o (2) Programas de Estudios? "))
            if respuesta in [1, 2]:
                break
            print("Respuesta inválida. Por favor, ingresa 1 para Opciones Ocupacionales o 2 para Programas de Estudios.")
        except ValueError:
            print("Por favor, ingresa un número válido.")

    if respuesta == 1:
        categoria = "Opciones Ocupacionales"
        print("\nCatálogo de Opciones Ocupacionales:")
        for idx, opcion in enumerate(opciones_ocupacionales, start=1):
            print(f"{idx}. {opcion}")
        while True:
            try:
                seleccion = int(input("Selecciona el número correspondiente a tu carrera: "))
                if 1 <= seleccion <= len(opciones_ocupacionales):
                    carrera = opciones_ocupacionales[seleccion - 1]
                    break
                else:
                    print("Número inválido. Inténtalo de nuevo.")
            except ValueError:
                print("Por favor, ingresa un número válido.")
    elif respuesta == 2:
        categoria = "Programas de Estudios"
        print("\nCatálogo de Programas de Estudios:")
        for idx, programa in enumerate(programas_estudio, start=1):
            print(f"{idx}. {programa}")
        while True:
            try:
                seleccion = int(input("Selecciona el número correspondiente a tu carrera: "))
                if 1 <= seleccion <= len(programas_estudio):
                    carrera = programas_estudio[seleccion - 1]
                    break
                else:
                    print("Número inválido. Inténtalo de nuevo.")
            except ValueError:
                print("Por favor, ingresa un número válido.")

    # Agregar respuesta al DataFrame usando pd.concat
    nuevo_dato = pd.DataFrame([{"Estudiante": f"Estudiante {estudiante}",
                                "Categoría": categoria,
                                "Carrera": carrera}])
    data = pd.concat([data, nuevo_dato], ignore_index=True)

# Resumen de estadísticas
resumen = data["Categoría"].value_counts()

print("\n--- Resumen de la Encuesta ---")
print(resumen)

# Visualización de resultados
plt.figure(figsize=(8, 6))
sns.barplot(x=resumen.index, y=resumen.values, palette="pastel")
plt.title("Preferencias de Estudio de los Estudiantes")
plt.ylabel("Número de Estudiantes")
plt.xlabel("Categorías")
plt.show()

# Preparar datos para el modelo
data["Categoría_Num"] = data["Categoría"].map({
    "Opciones Ocupacionales": 0,
    "Programas de Estudios": 1
})

X = pd.get_dummies(data["Carrera"])  # Convertir carreras a variables binarias
y = data["Categoría_Num"]

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar el modelo SVM
modelo_svm = SVC(kernel="linear", probability=True)
modelo_svm.fit(X_train, y_train)

# Evaluar el modelo
y_pred = modelo_svm.predict(X_test)
print("\n--- Resultados del Modelo SVM ---")
print(f"Exactitud: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, target_names=["Opciones Ocupacionales", "Programas de Estudios"]))

# Predicción con nueva carrera
nueva_carrera = input("\nIngresa una carrera para predecir su categoría (por ejemplo, 'Cocina'): ")
if nueva_carrera in opciones_ocupacionales + programas_estudio:
    nueva_carrera_vector = pd.DataFrame([nueva_carrera], columns=["Carrera"])
    nueva_carrera_vector = pd.get_dummies(nueva_carrera_vector)
    nueva_carrera_vector = nueva_carrera_vector.reindex(columns=X.columns, fill_value=0)
    prediccion = modelo_svm.predict(nueva_carrera_vector)
    categoria_predicha = "Programas de Estudios" if prediccion[0] == 1 else "Opciones Ocupacionales"
    print(f"La carrera '{nueva_carrera}' se predice como: {categoria_predicha}.")
else:
    print("La carrera ingresada no se encuentra en el catálogo.")
