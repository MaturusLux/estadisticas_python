# 01 Regresion Lineal - Precio Hectáreas de Lote
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Datos: Tamaño de Lote (Has) vs Precio (miles U$)
# Datos manuales en Arrays
X_lineal = np.array([50,80,100,120,150,180,200,250,300,500]).reshape(-1, 1)
y_lineal = np.array([50,75,98,115,150,160,170,235,245,420])

# Crear y entrenar el modelo
modelo_lineal = LinearRegression()
modelo_lineal.fit(X_lineal, y_lineal)

# Predicción para un lote de 90 Has
nuevo_lote = np.array([[90]])
precio_predicho = modelo_lineal.predict(nuevo_lote)

print("=== REGRESIÓN LINEAL ===")
print(f"Ecuación: Precio = {modelo_lineal.intercept_:.2f} + {modelo_lineal.coef_[0]:.2f} × Tamaño")
print(f"Predicción para 90 Has: ${precio_predicho[0]:.2f} miles")

# Gráfico
plt.scatter(X_lineal, y_lineal, color='blue', label='Datos reales')
plt.plot(X_lineal, modelo_lineal.predict(X_lineal), color='red', label='Línea de regresión')
plt.xlabel('Tamaño (Has)')
plt.ylabel('Precio (miles U$)')
plt.title('Regresión Lineal: Tamaño Lote vs Precio')
plt.legend()
plt.show()
