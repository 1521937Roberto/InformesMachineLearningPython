# Importar las bibliotecas necesarias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, r2_score

# Crear un conjunto de datos ficticio
data = {
    'corte_ensamblaje': [26, 20, 15],  # Estudiantes en Corte y Ensamblaje
    'panaderia_pasteleria': [48, 35, 40],  # Estudiantes en Panadería y Pastelería
    'apoyo_administrativo': [32, 30, 25],  # Estudiantes en Apoyo Administrativo
    'electricidad': [45, 55, 50],  # Estudiantes en Electricidad
    'peluqueria_basica': [122, 130, 125],  # Estudiantes en Peluquería Básica
    'computacion': [70, 75, 72]  # Estudiantes en Computación
}

# Convertir los datos a un DataFrame
df = pd.DataFrame(data)

# Definir las variables independientes (X) y dependiente (y)
X = df[['corte_ensamblaje', 'panaderia_pasteleria', 'apoyo_administrativo']]  # Programas de estudios
y = df[['electricidad', 'peluqueria_basica', 'computacion']]  # Opciones ocupacionales

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de regresión lineal múltiple
model = LinearRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

# Realizar predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo con métricas como el error cuadrático medio (MSE) y el coeficiente de determinación (R^2)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R²: {r2}')

# 1. Gráfico de barras comparando los valores reales y predichos

# Graficar la comparación de los valores reales y predichos para cada opción ocupacional
bar_width = 0.35
index = np.arange(len(y_test))

plt.figure(figsize=(10, 6))

# Graficar para Electricidad
plt.bar(index, y_test['electricidad'], bar_width, label='Real Electricidad', color='blue')
plt.bar(index + bar_width, y_pred[:, 0], bar_width, label='Predicción Electricidad', color='red')

# Graficar para Peluquería Básica
plt.bar(index + 2 * bar_width, y_test['peluqueria_basica'], bar_width, label='Real Peluquería Básica', color='green')
plt.bar(index + 3 * bar_width, y_pred[:, 1], bar_width, label='Predicción Peluquería Básica', color='orange')

# Graficar para Computación
plt.bar(index + 4 * bar_width, y_test['computacion'], bar_width, label='Real Computación', color='purple')
plt.bar(index + 5 * bar_width, y_pred[:, 2], bar_width, label='Predicción Computación', color='yellow')

plt.xlabel('Instancia de prueba')
plt.ylabel('Número de estudiantes')
plt.title('Comparación entre valores reales y predichos')
plt.legend()
plt.show()

# 2. Gráfico 3D de la relación entre los programas de estudios y las opciones ocupacionales

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Para graficar las relaciones entre Corte y Ensamblaje, Panadería y Apoyo Administrativo
# con las tres opciones ocupacionales
ax.scatter(X_test['corte_ensamblaje'], X_test['panaderia_pasteleria'], X_test['apoyo_administrativo'],
           c='blue', label='Programas de estudios', marker='o')

# Graficar las predicciones
ax.scatter(y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], c='red', label='Predicciones', marker='^')

ax.set_xlabel('Corte y Ensamblaje')
ax.set_ylabel('Panadería y Pastelería')
ax.set_zlabel('Apoyo Administrativo')
ax.set_title('Gráfico 3D de Programas de Estudios vs Opciones Ocupacionales')

plt.legend()
plt.show()
