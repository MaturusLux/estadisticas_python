# 03 Regresion Logística - Dosis Agroquímico Efectiva
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Datos: Dosis de Paraquat Lote X "Efectiva (1)" vs "No Efectiva (0)"
X_logistica = np.array([0, 0.3, 0.5, 0.7, 0.9, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 3, 3.2, 3.4, 3.6]).reshape(-1, 1)
y_logistica = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # 0=Reprobado, 1=Aprobado

# Crear y entrenar el modelo
modelo_logistica = LogisticRegression()
modelo_logistica.fit(X_logistica, y_logistica)

# Predicción para 1.5 horas de estudio
horas_estudio = np.array([[1.5]])
probabilidad = modelo_logistica.predict_proba(horas_estudio)

print("\n=== REGRESIÓN LOGÍSTICA ===")
print(f"Coeficiente (β₁): {modelo_logistica.coef_[0][0]:.2f}")
print(f"Intercepto (β₀): {modelo_logistica.intercept_[0]:.2f}")
print(f"Probabilidad de Eficiencia con 1.5 Litros: {probabilidad[0][1]*100:.1f}%")

# Gráfico de la curva sigmoide
X_test = np.linspace(0, 4, 100).reshape(-1, 1)
y_prob = modelo_logistica.predict_proba(X_test)[:, 1]

plt.scatter(X_logistica, y_logistica, color='blue', label='Datos reales')
plt.plot(X_test, y_prob, color='red', label='Probabil.(sigmoide)')
plt.xlabel('Dosis x Ha')
plt.ylabel('Probabilidad de eficiencia')
plt.title('Regresión Logística: Dosis x ha vs Probabilidad de exito')
plt.legend()
plt.show()
