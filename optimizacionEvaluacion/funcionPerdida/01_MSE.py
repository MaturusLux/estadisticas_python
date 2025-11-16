import numpy as np
# Error Cuadrático Medio/Mean Square Error
# Datos de ejemplo
y_real = np.array([3, -0.5, 2, 7])          # Valores reales
y_predicho = np.array([2.5, 0.0, 2, 8])     # Valores predichos por el modelo

print("Valores reales:", y_real)
print("Valores predichos:", y_predicho)

# 1. Calcular los errores individuales
errores = y_real - y_predicho
print("\n1. Errores individuales:", errores)

# 2. Elevar al cuadrado cada error
errores_cuadrados = errores ** 2
print("2. Errores al cuadrado:", errores_cuadrados)

# 3. Sumar todos los errores cuadrados
suma_errores_cuadrados = np.sum(errores_cuadrados)
print("3. Suma de errores cuadrados:", suma_errores_cuadrados)

# 4. Calcular el promedio (media)
n = len(y_real)
mse = suma_errores_cuadrados / n
print("4. Número de muestras (n):", n)
print("5. ECM final:", mse)

# Versión compacta en una línea
mse_compacto = np.mean((y_real - y_predicho) ** 2)
print("\nECM (versión compacta):", mse_compacto)
