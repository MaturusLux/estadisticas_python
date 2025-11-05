import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
# Crecimiento Exponencial Sumatoria Sucursales
# 1. Crear datos de ejemplo, sintéticos
np.random.seed(42)
x = np.linspace(0, 10, 100)
lambda_ = np.exp(0.5 + 0.3 * x)
y = np.random.poisson(lambda_)

# 2. Ajustar el modelo de regresión de Poisson
X = sm.add_constant(x)
modelo = sm.GLM(y, X, family=sm.families.Poisson())
resultados = modelo.fit()
print(resultados.summary())

# 3. Predecir y visualizar los resultados (incluye bandas de confianza)
y_predicho = resultados.predict(X)

# Obtener intervalos de confianza para la predicción (response scale)
pred = resultados.get_prediction(X)
pred_df = pred.summary_frame(alpha=0.05)  # 95% CI
# pred_df contiene columnas: mean, mean_se, mean_ci_lower, mean_ci_upper (en
# escala de respuesta)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', alpha=0.6, label='Datos observados')
plt.plot(x, y_predicho, color='red', linewidth=2, label='Predicción (media)')
plt.fill_between(x, pred_df['mean_ci_lower'], pred_df['mean_ci_upper'], color='red', alpha=0.2, label='IC 95%')
plt.title('Regresión de Poisson y visualización con Matplotlib [Crecimiento Exponencial Sumatoria Sucursales]')
plt.xlabel('Variable independiente (x)[Años]')
plt.ylabel('Conteo de eventos (y)[Ventas Agroquímicos Globales]')
plt.legend()
plt.grid(True)
plt.show()
