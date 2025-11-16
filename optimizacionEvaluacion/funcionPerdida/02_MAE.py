import numpy as np
# Error Absoluto Medio
# Datos de ejemplo (los mismos que usamos para ECM)
y_real = np.array([3, -0.5, 2, 7])          # Valores reales
y_predicho = np.array([2.5, 0.0, 2, 8])     # Valores predichos por el modelo

print("Valores reales:", y_real)
print("Valores predichos:", y_predicho)

# 1. Calcular los errores individuales
errores = y_real - y_predicho
print("\n1. Errores individuales:", errores)

# 2. Tomar el valor absoluto de cada error
errores_absolutos = np.abs(errores)
print("2. Errores absolutos:", errores_absolutos)

# 3. Sumar todos los errores absolutos
suma_errores_absolutos = np.sum(errores_absolutos)
print("3. Suma de errores absolutos:", suma_errores_absolutos)

# 4. Calcular el promedio (media)
n = len(y_real)
mae = suma_errores_absolutos / n
print("4. Número de muestras (n):", n)
print("5. MAE final:", mae)

# Versión compacta en una línea
mae_compacto = np.mean(np.abs(y_real - y_predicho))
print("\nMAE (versión compacta):", mae_compacto)
