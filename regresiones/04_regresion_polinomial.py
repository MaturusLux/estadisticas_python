# 04 Regresion Polinomial - Volatilidad Precio Futuros
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Datos no lineales, sintéticos.
np.random.seed(42)
X = np.linspace(0, 12, 100)
y = 2 + 1.5*X - 0.3*X**2 + 0.03*X**3 + np.random.normal(0, 2, 100)

X = X.reshape(-1, 1)

# Transformación polinomial (grado 3)
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Modelo lineal con características polinomiales
modelo_poly = LinearRegression()
modelo_poly.fit(X_poly, y)

# Predicciones
X_test = np.linspace(0, 12, 50).reshape(-1, 1)
X_test_poly = poly.transform(X_test)
y_pred = modelo_poly.predict(X_test_poly)

print("=== REGRESIÓN POLINOMIAL / VOLATILIDAD PORCENTUAL ===")
print(f"Coeficientes: {modelo_poly.coef_}")
print(f"Intercepto: {modelo_poly.intercept_:.2f}")

plt.scatter(X, y, alpha=0.7, label='Datos reales')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regre. Polinomial (grado 3)')
plt.title("Volatilidad Futuro Grano 'X1'(Acelera Diciembre) - REGRESION POLINOMIAL")
plt.xlabel('X (Enero->Diciembre)')
plt.ylabel('y (Porcentaje Variación')
plt.legend()
plt.show()
