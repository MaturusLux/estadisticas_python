import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 1. Crear datos de ejemplo con una tendencia no lineal
np.random.seed(42)
n_puntos = 150
x = np.linspace(0, 10, n_puntos)
# Relación no lineal: y = sin(x) + un poco de ruido
y = np.sin(x) + np.random.normal(0, 0.2, n_puntos)

# 2. Calcular la regresión LOWESS para diferentes fracciones de suavizado
# Fracción pequeña: menos suavizado, más detalle
lowess_frac_menor = sm.nonparametric.lowess(y, x, frac=0.2)
# Fracción más grande: más suavizado, menos detalle
lowess_frac_mayor = sm.nonparametric.lowess(y, x, frac=0.6)

# 3. Visualizar los resultados con Matplotlib
plt.figure(figsize=(12, 8))

# Gráfico de dispersión de los datos originales
plt.scatter(x, y, label='Datos originales con ruido', color='gray', alpha=0.6)

# Trazar la línea LOWESS con una fracción menor
plt.plot(lowess_frac_menor[:, 0], lowess_frac_menor[:, 1], 
         label=f'LOWESS (frac=0.2)', color='blue', linewidth=2)

# Trazar la línea LOWESS con una fracción mayor
plt.plot(lowess_frac_mayor[:, 0], lowess_frac_mayor[:, 1], 
         label=f'LOWESS (frac=0.6)', color='red', linewidth=2, linestyle='--')

plt.title('Variación Porcentual Precio X - Regresión LOWESS con diferentes parámetros')
plt.xlabel('Años - Variable Independiente (x)')
plt.ylabel('Porcentaje % -Variable Dependiente (y)')
plt.legend()
plt.grid(True)
plt.show()
