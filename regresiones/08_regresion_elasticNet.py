import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Fijar semilla para reproducibilidad
np.random.seed(42)

# ================================
# 1. Generación de datos sintéticos
# ================================
n_samples = 500

# Variables agropecuarias (características)
# Cada fila = un lote de soja
X = np.zeros((n_samples, 8))

# 1. Precipitación acumulada en floración (mm)
X[:, 0] = np.random.normal(200, 50, n_samples)
X[:, 0] = np.clip(X[:, 0], 50, 400)  # rango realista

# 2. Temperatura media durante llenado de grano (°C)
X[:, 1] = np.random.normal(24, 3, n_samples)
X[:, 1] = np.clip(X[:, 1], 18, 32)

# 3. Densidad de siembra (plantas/ha * 1000)
X[:, 2] = np.random.uniform(250, 400, n_samples)

# 4. Aplicaciones de fungicida (número de aplicaciones)
X[:, 3] = np.random.poisson(2, n_samples)
X[:, 3] = np.clip(X[:, 3], 0, 5)

# 5. Aplicaciones de insecticida
X[:, 4] = np.random.poisson(1.5, n_samples)
X[:, 4] = np.clip(X[:, 4], 0, 4)

# 6. Índice de malezas (0 = bajo, 1 = alto; escala continua)
X[:, 5] = np.random.beta(2, 5, n_samples)  # sesgado a bajo

# 7. Estrés hídrico (días sin riego/lluvia significativa en crítico)
X[:, 6] = np.random.exponential(5, n_samples)
X[:, 6] = np.clip(X[:, 6], 0, 20)

# 8. Fertilización con N (kg/ha)
X[:, 7] = np.random.uniform(0, 30, n_samples)

# ================================
# 2. Generar variable objetivo: Merma de calidad (%)
# ================================
# La merma depende de:
# - Alta temperatura + baja lluvia → más daño
# - Pocas aplicaciones de fungicida → más enfermedades
# - Muchas malezas o estrés → calidad baja
# - Exceso de N → desbalance

merma = (
    0.02 * X[:, 6] +                     # estrés hídrico
    0.05 * (32 - X[:, 1]) +              # calor extremo (más daño si T > 28)
    0.8 * (2 - X[:, 3]) +                # menos fungicida → más merma
    1.2 * X[:, 5] +                      # malezas
    0.03 * np.abs(X[:, 7] - 15) +        # desbalance N
    2.0 +                                # base
    np.random.normal(0, 0.8, n_samples)  # ruido
)

# Asegurar que la merma esté entre 0% y 15%
merma = np.clip(merma, 0, 15)

y = merma  # objetivo: porcentaje de merma

# ================================
# 3. Preparar datos para el modelo
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalar características (importante para ElasticNet)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================================
# 4. Entrenar modelo ElasticNet
# ================================
# alpha = fuerza total de regularización
# l1_ratio = proporción L1 (0 = Ridge, 1 = Lasso)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.7, random_state=42, max_iter=10000)
elastic.fit(X_train_scaled, y_train)

# Predicciones
y_train_pred = elastic.predict(X_train_scaled)
y_test_pred = elastic.predict(X_test_scaled)

# Métricas
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("=== MODELO ELASTICNET: PREDICCIÓN DE MERMA EN SOJA ===")
print(f"R² en entrenamiento: {train_r2:.3f}")
print(f"R² en prueba:        {test_r2:.3f}")
print(f"RMSE en prueba:      {test_rmse:.3f} %")

# Validación cruzada
cv_scores = cross_val_score(elastic, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"R² CV (5-fold):      {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ================================
# 5. Interpretación: Coeficientes
# ================================
feature_names = [
    "Precip. Floración (mm)",
    "Temp. Llenado (°C)",
    "Dens. Siembra (mil/ha)",
    "Aplic. Fungicida",
    "Aplic. Insecticida",
    "Índice Malezas",
    "Estrés Hídrico (días)",
    "Fert. N (kg/ha)"
]

coef = elastic.coef_
importancia = np.abs(coef)

# Mostrar coeficientes
print("\n=== COEFICIENTES DEL MODELO (ElasticNet) ===")
for name, c, imp in zip(feature_names, coef, importancia):
    print(f"{name:<25}: {c:7.3f}  (|coef| = {imp:.3f})")

# Variables irrelevantes (coef ≈ 0) son eliminadas por L1
variables_seleccionadas = [name for name, c in zip(feature_names, coef) if abs(c) > 0.01]
print(f"\nVariables seleccionadas ({len(variables_seleccionadas)}):")
for v in variables_seleccionadas:
    print(f"  - {v}")

# ================================
# 6. Visualización
# ================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 6.1. Predicción vs Real (conjunto de prueba)
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.7, color='darkgreen')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Merma Real (%)')
axes[0, 0].set_ylabel('Merma Predicha (%)')
axes[0, 0].set_title(f'Predicción vs Real (R² = {test_r2:.2f})')
axes[0, 0].grid(True, alpha=0.3)

# 6.2. Importancia de variables (coeficientes)
y_pos = np.arange(len(feature_names))
axes[0, 1].barh(y_pos, coef, color=np.where(coef > 0, 'red', 'blue'), alpha=0.8)
axes[0, 1].set_yticks(y_pos)
axes[0, 1].set_yticklabels(feature_names)
axes[0, 1].set_xlabel('Coeficiente')
axes[0, 1].set_title('Impacto de Variables en Merma')
axes[0, 1].axvline(0, color='black', linewidth=0.8)
axes[0, 1].grid(True, axis='x', alpha=0.3)

# 6.3. Distribución de merma
axes[1, 0].hist(y, bins=25, color='orange', alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Merma de Calidad (%)')
axes[1, 0].set_ylabel('Frecuencia')
axes[1, 0].set_title('Distribución de Mermas en Lotes')
axes[1, 0].grid(True, alpha=0.3)

# 6.4. Residuos
residuos = y_test - y_test_pred
axes[1, 1].scatter(y_test_pred, residuos, alpha=0.7, color='purple')
axes[1, 1].axhline(0, color='red', linestyle='--')
axes[1, 1].set_xlabel('Merma Predicha (%)')
axes[1, 1].set_ylabel('Residuos')
axes[1, 1].set_title('Análisis de Residuos')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ================================
# 7. Ejemplo de predicción para un nuevo lote
# ================================
nuevo_lote = np.array([[220, 26, 320, 3, 2, 0.3, 4, 12]])  # valores típicos
nuevo_lote_scaled = scaler.transform(nuevo_lote)
merma_predicha = elastic.predict(nuevo_lote_scaled)[0]

print("\n" + "="*60)
print("EJEMPLO: PREDICCIÓN PARA UN NUEVO LOTE")
print("="*60)
print(f"Condiciones del lote:")
print(f"  - Precip. floración: 220 mm")
print(f"  - Temp. llenado:     26°C")
print(f"  - Dens. siembra:     320 mil/ha")
print(f"  - Fungicidas:        3 aplic.")
print(f"  - Insecticidas:      2 aplic.")
print(f"  - Malezas:           0.3 (bajo)")
print(f"  - Estrés hídrico:    4 días")
print(f"  - Fertilización N:   12 kg/ha")
print(f"\n Merma de calidad predicha: {merma_predicha:.2f} %")
