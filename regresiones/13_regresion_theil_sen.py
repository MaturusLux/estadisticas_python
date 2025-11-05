import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import TheilSenRegressor, LinearRegression

# 1. Generar datos de ejemplo con valores atípicos (outliers)
np.random.seed(42)  # Para reproducibilidad
n_samples = 200

# Datos con una relación lineal
x = np.random.randn(n_samples)+3
true_w = 3.0  # Pendiente verdadera
true_c = 12.0  # Intercepto verdadero
noise = 0.5 * np.random.randn(n_samples)
y = true_w * x + true_c + noise

# Añadir valores atípicos significativos
y[-15:] += -8 * x[-1:]
X = x[:, np.newaxis] # Reshape para scikit-learn

# 2. Inicializar y entrenar el modelo de Theil-Sen
theil_sen = TheilSenRegressor(random_state=42)
theil_sen.fit(X, y)

# 3. Realizar predicciones con el modelo de Theil-Sen
y_pred_theil_sen = theil_sen.predict(X)

# Opcionalmente, entrenar y predecir con una regresión OLS para comparar
ols = LinearRegression()
ols.fit(X, y)
y_pred_ols = ols.predict(X)

# 4. Visualizar los resultados
plt.figure(figsize=(10, 7))
plt.scatter(x, y, color='blue', label='Datos con Outliers - Lote sin Aplicar (atípico o erroneo)')
plt.plot(x, y_pred_theil_sen, color='orange', linewidth=2, label='Regresión de Theil-Sen')
plt.plot(x, y_pred_ols, color='red', linestyle='--', linewidth=2, label='Regresión OLS (sensible a outliers)')

plt.xlabel("Lotes - Variable Independiente (x)")
plt.ylabel("Conteo Plaga - Variable Dependiente (y)")
plt.title("Monitoreo x Lote c/Aplicación - Regresión de Theil-Sen vs. OLS con Outliers")
plt.legend()
plt.grid(True)
plt.show()

# 5. Imprimir los coeficientes (pendiente e intercepto)
print("--- Coeficientes del modelo de Theil-Sen ---")
print(f"Pendiente (coef_): {theil_sen.coef_[0]:.4f}")
print(f"Intercepto (intercept_): {theil_sen.intercept_:.4f}")
print("\n--- Coeficientes del modelo OLS (comparación) ---")
print(f"Pendiente (coef_): {ols.coef_[0]:.4f}")
print(f"Intercepto (intercept_): {ols.intercept_:.4f}")
