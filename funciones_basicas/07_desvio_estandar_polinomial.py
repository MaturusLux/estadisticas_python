import numpy as np
import matplotlib.pyplot as plt

# 1. Datos de ejemplo
np.random.seed(42)
x = np.linspace(1, 12, 100)
# Creamos datos que siguen una curva polinomial (grado 3) con ruido.
y = 0.25 * x**3 - 2.5 * x**2 + 5 * x + 25 + np.random.normal(0, 10, 100)

# 2. Calcular la regresión polinomial
# El tercer argumento de np.polyfit es el grado del polinomio.
grado_polinomio = 3
coeffs = np.polyfit(x, y, grado_polinomio)

# 3. Crear el modelo polinomial con los coeficientes
# np.poly1d crea una función polinomial a partir de los coeficientes.
poly_model = np.poly1d(coeffs)

# 4. Calcular los valores predichos y los residuos
y_pred = poly_model(x)
residuos = y - y_pred

# 5. Calcular la varianza y la desviación estándar de los residuos
varianza_residuos = np.var(residuos)
std_dev_residuos = np.sqrt(varianza_residuos)

# 6. Configurar el gráfico
plt.figure(figsize=(12, 8))
plt.scatter(x, y, label='Datos originales (Demanda Día', color='blue', alpha=0.7)
plt.plot(x, y_pred, color='red', linewidth=2, label=f'Regresión Polinomial (Grado {grado_polinomio})')

# 7. Visualizar las bandas de desviación estándar
# Se usan las mismas técnicas que en el caso lineal para mostrar la dispersión.
plt.fill_between(x, y_pred - std_dev_residuos, y_pred + std_dev_residuos, 
                 color='orange', alpha=0.3, label='±1 Desviación Estándar')
plt.fill_between(x, y_pred - 2 * std_dev_residuos, y_pred + 2 * std_dev_residuos, 
                 color='yellow', alpha=0.2, label='±2 Desviaciones Estándar')

# 8. Añadir etiquetas, título y leyenda
plt.title('Demanda de Agroquimico X1--Regresión Polinomial con Bandas de Desviación Estándar', fontsize=16)
plt.xlabel('Mes - Variable Independiente (x)', fontsize=12)
plt.ylabel('Litros x Miles - Variable Dependiente (y)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 9. Imprimir los resultados
print(f"Coeficientes del polinomio: {coeffs}")
print(f"Varianza de los residuos: {varianza_residuos:.2f}")
print(f"Desviación estándar de los residuos: {std_dev_residuos:.2f}")
