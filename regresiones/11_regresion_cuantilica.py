import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import QuantileRegressor
# Gastos no previstos en Campaña.
# Datos con heterocedasticidad (varianza no constante)
np.random.seed(42)
X_quant = np.random.uniform(0, 12, 100)
# Varianza aumenta con X
y_quant = 2 + 1.5 * X_quant + np.random.normal(0, 0.5 * X_quant, 100)

X_quant = X_quant.reshape(-1, 1)

# Modelos para diferentes cuantiles
quantiles = [0.2, 0.5, 0.8]
colors = ['blue', 'red', 'green']
labels = ['Q20-Soporte Gastos', 'Mediana (Q50)-Promedio Gastos', 'Q90-Techo Gastos']

plt.figure(figsize=(10, 6))
plt.scatter(X_quant, y_quant, alpha=0.5, label='Datos Gastos no Previstos por Año')

for q, color, label in zip(quantiles, colors, labels):
    # Usando scikit-learn (más nuevo)
    modelo_quant = QuantileRegressor(quantile=q, alpha=0)
    modelo_quant.fit(X_quant, y_quant)
    
    # Predicciones
    X_test_quant = np.linspace(0, 12, 50).reshape(-1, 1)
    y_pred_quant = modelo_quant.predict(X_test_quant)
    
    plt.plot(X_test_quant, y_pred_quant, color=color, linewidth=2, label=label)
    
    print(f"\n=== REGRESIÓN CUANTÍLICA -GASTOSS NO PREVISTOS({label}) ===")
    print(f"Intercepto: {modelo_quant.intercept_:.3f}")
    print(f"Coeficiente: {modelo_quant.coef_[0]:.3f}")

plt.xlabel('X')
plt.ylabel('y')
plt.title('Regresión Cuantílica - Rango Gastos No Previsibles x Cosecha')
plt.legend()
plt.show()
