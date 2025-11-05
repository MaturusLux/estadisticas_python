# 02 Regresión Lineal Múltiple - Precio Hectáreas de Lote
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# 1. Crear un conjunto de datos ficticio con 3 variables agrícolas
# Datos Sintéticos con numpy.random.
np.random.seed(42)
n_lotes = 100

# Variables independientes
horas_labranza = np.random.uniform(5, 40, n_lotes)  # Horas de labranza y monitoreo
aplicacion_productos = np.random.uniform(10, 100, n_lotes) # Unidades de productos aplicados

# Variable dependiente (rinde final)
# rinde = 20 * horas_labranza + 15 * aplicacion_productos + ruido
rinde_final = (20 * horas_labranza + 15 * aplicacion_productos +
                np.random.normal(0, 100, n_lotes))

# Asegurar que el rinde esté en el rango de 2000 a 4000 kgs
rinde_final = np.clip(rinde_final, 2000, 4000)

# 2. Preparar los datos para el modelo
X = np.column_stack((horas_labranza, aplicacion_productos))
y = rinde_final

# 3. Crear y entrenar el modelo de regresión lineal múltiple
modelo = LinearRegression()
modelo.fit(X, y)

# 4. Predicciones para el plano de regresión
# Crear una malla para el plano de regresión
x1_surf, x2_surf = np.meshgrid(np.linspace(min(horas_labranza), max(horas_labranza), 10), 
                                 np.linspace(min(aplicacion_productos), max(aplicacion_productos), 10))
plano_predicho = modelo.intercept_ + modelo.coef_[0] * x1_surf + modelo.coef_[1] * x2_surf

# 5. Visualizar los resultados en 3D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Graficar los datos reales como puntos de dispersión
ax.scatter(horas_labranza, aplicacion_productos, rinde_final, color='green', label='Datos reales de lotes', s=40)

# Graficar el plano de regresión
ax.plot_surface(x1_surf, x2_surf, plano_predicho, color='red', alpha=0.5)

# Etiquetas y título
ax.set_xlabel('Horas de Labranza y Monitoreo')
ax.set_ylabel('Aplicación de Productos (unidades)')
ax.set_zlabel('Rinde Final (kg por lote)')
ax.set_title('Regresión Lineal: Horas de Manejo y Productos vs. Rinde')
ax.view_init(elev=25, azim=50) # Ajustar el ángulo de visión

# Para que la leyenda se muestre correctamente
ax.legend([plt.Line2D([0],[0], color='green', marker='o', linestyle='', markersize=7),
           plt.Line2D([0],[0], color='red', marker='s', linestyle='', alpha=0.5, markersize=7)],
          ['Datos reales', 'Plano de regresión'])

plt.show()

# Imprimir los coeficientes e intercepto del modelo
print("--- Coeficientes del modelo de regresión ---")
print(f"Intercepto: {modelo.intercept_:.2f}")
print(f"Coeficiente para Horas de Labranza: {modelo.coef_[0]:.2f}")
print(f"Coeficiente para Aplicación de Productos: {modelo.coef_[1]:.2f}")
print("-" * 45)
