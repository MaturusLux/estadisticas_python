import numpy as np
import matplotlib.pyplot as plt

# 1. Datos de ejemplo
np.random.seed(42)  # Para reproducibilidad
x = np.linspace(0, 10, 50)
y = 2.5 * x + 5 + np.random.normal(0, 3, 50) # y = 2.5x + 5 + ruido

# 2. Calcular la regresión lineal
coeffs = np.polyfit(x, y, 1)
y_pred = coeffs[0] * x + coeffs[1]

# 3. Calcular la varianza y la desviación estándar de los residuos
residuos = y - y_pred
varianza_residuos = np.var(residuos)
std_dev_residuos = np.sqrt(varianza_residuos)

# 4. Configurar el gráfico
plt.figure(figsize=(12, 8))
plt.scatter(x, y, label='Datos originales', color='blue', alpha=0.7)
plt.plot(x, y_pred, color='red', linewidth=2, label='Línea de regresión')

# 5. Visualizar las bandas de desviación estándar
# La desviación estándar es la medida más común para visualizar la dispersión.
# Graficamos las bandas de 1 y 2 desviaciones estándar.
plt.fill_between(x, y_pred - std_dev_residuos, y_pred + std_dev_residuos, 
                 color='orange', alpha=0.3, label='±1 Desviación Estándar')
plt.fill_between(x, y_pred - 2 * std_dev_residuos, y_pred + 2 * std_dev_residuos, 
                 color='yellow', alpha=0.2, label='±2 Desviaciones Estándar')

# 6. Añadir etiquetas, título y leyenda
plt.title('Precio Has -Regresión Lineal con Bandas de Desviación Estándar', fontsize=16)
plt.xlabel('Has - Variable Independiente (x)', fontsize=12)
plt.ylabel('U$ por Ha - Variable Dependiente (y)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 7. Imprimir los resultados
print(f"Ecuación de la regresión: y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}")
print(f"Varianza de los residuos: {varianza_residuos:.2f}")
print(f"Desviación estándar de los residuos: {std_dev_residuos:.2f}")
