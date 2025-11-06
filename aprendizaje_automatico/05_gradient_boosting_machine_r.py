import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Boosting - Regressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# Generar datos sintéticos de suelos (2000-2024)
np.random.seed(42)
num_lotes = 2000

data = {
    'Año': np.random.randint(2000, 2025, num_lotes),
    'pH': np.round(np.random.uniform(5.0, 8.5, num_lotes), 1),
    'Materia_Organica': np.round(np.random.uniform(0.5, 6.0, num_lotes), 2),
    'Nitrogeno': np.round(np.random.uniform(10, 150, num_lotes), 0),
    'Fosforo': np.round(np.random.uniform(5, 80, num_lotes), 0),
    'Potasio': np.round(np.random.uniform(50, 300, num_lotes), 0),
    'Textura_Limoso': np.round(np.random.uniform(0.1, 0.8, num_lotes), 2),
    'Textura_Arenoso': np.round(np.random.uniform(0.1, 0.7, num_lotes), 2),
    'Textura_Arcilloso': np.round(np.random.uniform(0.1, 0.9, num_lotes), 2),
}

# Normalizar texturas para que sumen 1
textura_total = data['Textura_Limoso'] + data['Textura_Arenoso'] + data['Textura_Arcilloso']
data['Textura_Limoso'] /= textura_total
data['Textura_Arenoso'] /= textura_total
data['Textura_Arcilloso'] /= textura_total

# Crear DataFrame
df = pd.DataFrame(data)

# DEFINIR REGLAS PARA CULTIVOS ÓPTIMOS (basado en características reales)
def asignar_cultivo_optimo(row):
    pH = row['pH']
    materia_organica = row['Materia_Organica']
    nitrogeno = row['Nitrogeno']
    fosforo = row['Fosforo']
    limoso = row['Textura_Limoso']
    arenoso = row['Textura_Arenoso']

    # Soja - prefiere suelos bien drenados, pH neutro, buen nitrógeno
    if (6.0 <= pH <= 7.2 and materia_organica > 2.5 and
        nitrogeno > 60 and limoso > 0.4):
        return 'Soja'

    # Maíz - requiere buen fósforo, materia orgánica, pH ligeramente ácido a neutro
    elif (5.8 <= pH <= 7.0 and fosforo > 30 and
          materia_organica > 3.0 and limoso > 0.3):
        return 'Maíz'

    # Trigo - prefiere suelos arcillosos, pH neutro, buen nitrógeno
    elif (6.5 <= pH <= 7.5 and nitrogeno > 70 and
          row['Textura_Arcilloso'] > 0.4):
        return 'Trigo'

    # Sorgo - tolera suelos más secos y menos fértiles
    elif (pH >= 5.5 and materia_organica > 1.5 and
          arenoso > 0.3 and nitrogeno < 80):
        return 'Sorgo'

    # Algodón - prefiere suelos bien drenados, pH ligeramente ácido
    elif (5.8 <= pH <= 6.8 and arenoso > 0.4 and
          fosforo > 25 and materia_organica > 2.0):
        return 'Algodón'

    else:
        # Asignar basado en probabilidades si no cumple reglas específicas
        cultivos = ['Soja', 'Maíz', 'Trigo', 'Sorgo', 'Algodón']
        pesos = [0.25, 0.25, 0.2, 0.15, 0.15]
        return np.random.choice(cultivos, p=pesos)

# Aplicar reglas para asignar cultivos óptimos
df['Cultivo_Optimo'] = df.apply(asignar_cultivo_optimo, axis=1)

print("=== GRADIENT BOOSTING CLASSIFIER - PREDICCIÓN DE CULTIVOS ===")
print(f"Total de lotes: {len(df)}")
print("\nDistribución de cultivos óptimos:")
print(df['Cultivo_Optimo'].value_counts())

# Preparar datos para el modelo
features = ['pH', 'Materia_Organica', 'Nitrogeno', 'Fosforo', 'Potasio',
           'Textura_Limoso', 'Textura_Arenoso', 'Textura_Arcilloso']

X = df[features]
y = df['Cultivo_Optimo']

# Codificar variable objetivo
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Modelo Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    subsample=0.8
)

# Entrenar modelo
gb_classifier.fit(X_train, y_train)

# Predicciones
y_pred = gb_classifier.predict(X_test)
y_pred_proba = gb_classifier.predict_proba(X_test)

# Métricas
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo: {accuracy:.3f}")

# Reporte de clasificación
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# VISUALIZACIONES
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 1. Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax1)
ax1.set_title('Matriz de Confusión - Clasificación de Cultivos')
ax1.set_xlabel('Predicho')
ax1.set_ylabel('Real')

# 2. Importancia de características
feature_importance = gb_classifier.feature_importances_
feature_names = features
sorted_idx = np.argsort(feature_importance)

ax2.barh(range(len(sorted_idx)), feature_importance[sorted_idx], color='lightgreen')
ax2.set_yticks(range(len(sorted_idx)))
ax2.set_yticklabels([feature_names[i] for i in sorted_idx])
ax2.set_title('Importancia de Características del Suelo')
ax2.set_xlabel('Importancia')

# 3. Evolución del accuracy durante entrenamiento
train_accuracy = []
test_accuracy = []

for i, y_pred_stage in enumerate(gb_classifier.staged_predict(X_train)):
    train_accuracy.append(accuracy_score(y_train, y_pred_stage))

for i, y_pred_stage in enumerate(gb_classifier.staged_predict(X_test)):
    test_accuracy.append(accuracy_score(y_test, y_pred_stage))

ax3.plot(range(1, len(train_accuracy) + 1), train_accuracy, 'b-', label='Entrenamiento', alpha=0.7)
ax3.plot(range(1, len(test_accuracy) + 1), test_accuracy, 'r-', label='Prueba', alpha=0.7)
ax3.set_xlabel('Número de Árboles')
ax3.set_ylabel('Accuracy')
ax3.set_title('Evolución del Accuracy durante Entrenamiento')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Distribución de probabilidades por cultivo
prob_df = pd.DataFrame(y_pred_proba, columns=le.classes_)
prob_melted = prob_df.melt(var_name='Cultivo', value_name='Probabilidad')

sns.boxplot(data=prob_melted, x='Cultivo', y='Probabilidad', ax=ax4)
ax4.set_title('Distribución de Probabilidades de Predicción por Cultivo')
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# RECOMENDACIONES POR LOTE
print("\n" + "="*60)
print("RECOMENDACIONES DE CULTIVOS PARA LOTES DE PRUEBA")
print("="*60)

# Mostrar primeras 10 predicciones del test
test_indices = X_test.index[:10]
for idx in test_indices:
    lote_data = df.loc[idx, features]
    cultivo_real = df.loc[idx, 'Cultivo_Optimo']

    # Convert lote_data Series to DataFrame with feature names
    lote_data_df = pd.DataFrame([lote_data], columns=features)

    cultivo_pred = le.inverse_transform([gb_classifier.predict(lote_data_df)[0]])[0]
    proba = np.max(gb_classifier.predict_proba(lote_data_df))

    print(f"\nLote {idx}:")
    print(f"  Características: pH={lote_data['pH']}, MO={lote_data['Materia_Organica']:.2f}%, "
          f"N={lote_data['Nitrogeno']} ppm, P={lote_data['Fosforo']} ppm")
    print(f"  Textura: Limoso={lote_data['Textura_Limoso']:.2f}, "
          f"Arenoso={lote_data['Textura_Arenoso']:.2f}, "
          f"Arcilloso={lote_data['Textura_Arcilloso']:.2f}")
    print(f"  Cultivo Real: {cultivo_real}")
    print(f"  Cultivo Recomendado: {cultivo_pred} (confianza: {proba:.2f})")


# EJEMPLO 2: GRADIENT BOOSTING REGRESSOR - RENDIMIENTO ESPERADO
print("\n" + "="*60)
print("GRADIENT BOOSTING REGRESSOR - PREDICCIÓN DE RENDIMIENTO")
print("="*60)

# Agregar variable de rendimiento esperado (kg/ha) basado en características del suelo y cultivo
def calcular_rendimiento_esperado(row):
    cultivo = row['Cultivo_Optimo']
    base_rendimiento = {
        'Soja': 3500,
        'Maíz': 5500,
        'Trigo': 2200,
        'Sorgo': 2500,
        'Algodón': 2100
    }

    rendimiento_base = base_rendimiento.get(cultivo, 2000) # Default base yield

    # Ajustes por calidad del suelo (more conservative adjustments)
    ajuste_pH = 1.0
    if cultivo in ['Soja', 'Trigo'] and 6.5 <= row['pH'] <= 7.0:
        ajuste_pH = 1.05 # Reduced adjustment
    elif cultivo in ['Maíz', 'Sorgo'] and 6.0 <= row['pH'] <= 6.8:
        ajuste_pH = 1.03 # Reduced adjustment

    ajuste_mo = 1.0 + (row['Materia_Organica'] - 2.5) * 0.05 # Reduced impact
    ajuste_n = 1.0 + (row['Nitrogeno'] - 50) * 0.002 # Reduced impact
    ajuste_p = 1.0 + (row['Fosforo'] - 20) * 0.005 # Reduced impact

    # Ajuste por textura (more conservative adjustments)
    ajuste_textura = 1.0
    if cultivo in ['Soja', 'Algodón'] and row['Textura_Limoso'] > 0.4:
        ajuste_textura = 1.05 # Reduced adjustment
    elif cultivo == 'Trigo' and row['Textura_Arcilloso'] > 0.3:
        ajuste_textura = 1.03 # Reduced adjustment

    rendimiento_final = rendimiento_base * ajuste_pH * ajuste_mo * ajuste_n * ajuste_p * ajuste_textura

    # Añadir variabilidad
    rendimiento_final *= np.random.uniform(0.95, 1.05) # Reduced variability

    # print(f"Cultivo: {cultivo}, Base: {rendimiento_base}, pH_adj: {ajuste_pH:.2f}, MO_adj: {ajuste_mo:.2f}, N_adj: {ajuste_n:.2f}, P_adj: {ajuste_p:.2f}, Text_adj: {ajuste_textura:.2f}, Final: {rendimiento_final:.0f}") # Uncomment for debugging

    return int(rendimiento_final)

# Use .copy() to avoid SettingWithCopyWarning
df_reg = df.copy()
df_reg['Rendimiento_Esperado'] = df_reg.apply(calcular_rendimiento_esperado, axis=1)

print(f"Rendimiento promedio por cultivo:")
print(df_reg.groupby('Cultivo_Optimo')['Rendimiento_Esperado'].describe())

# Preparar datos para regresión
X_reg = df_reg[features + ['Cultivo_Optimo']].copy() # Use .copy()
# Codificar cultivo como variable numérica para regresión
X_reg['Cultivo_Cod'] = le.transform(X_reg['Cultivo_Optimo'])
X_reg = X_reg[features + ['Cultivo_Cod']]
y_reg = df_reg['Rendimiento_Esperado']

# Dividir datos
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Modelo Gradient Boosting Regressor
gb_regressor = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=4,
    min_samples_split=15,
    min_samples_leaf=8,
    random_state=42,
    subsample=0.8,
    loss='huber'  # Robust to outliers
)

# Entrenar modelo
gb_regressor.fit(X_train_reg, y_train_reg)

# Predicciones
y_pred_reg = gb_regressor.predict(X_test_reg)

# Métricas de regresión
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test_reg, y_pred_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"\nMétricas de Regresión:")
print(f"MAE (Error Absoluto Medio): {mae:.0f} kg/ha")
print(f"MSE (Error Cuadrático Medio): {mse:.0f}")
print(f"R² (Coeficiente de determinación): {r2:.3f}")

# VISUALIZACIONES REGRESIÓN
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 1. Predicciones vs Valores reales
ax1.scatter(y_test_reg, y_pred_reg, alpha=0.6, color='blue')
ax1.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
ax1.set_xlabel('Rendimiento Real (kg/ha)')
ax1.set_ylabel('Rendimiento Predicho (kg/ha)')
ax1.set_title('Predicciones vs Valores Reales')
ax1.grid(True, alpha=0.3)

# 2. Importancia de características para regresión
feature_importance_reg = gb_regressor.feature_importances_
feature_names_reg = features + ['Cultivo_Cod']
sorted_idx_reg = np.argsort(feature_importance_reg)

ax2.barh(range(len(sorted_idx_reg)), feature_importance_reg[sorted_idx_reg], color='lightcoral')
ax2.set_yticks(range(len(sorted_idx_reg)))
ax2.set_yticklabels([feature_names_reg[i] for i in sorted_idx_reg])
ax2.set_title('Importancia de Características - Predicción de Rendimiento')
ax2.set_xlabel('Importancia')

# 3. Evolución del error durante entrenamiento
train_errors = []
test_errors = []

for i, y_pred_stage in enumerate(gb_regressor.staged_predict(X_train_reg)):
    train_errors.append(mean_absolute_error(y_train_reg, y_pred_stage))

for i, y_pred_stage in enumerate(gb_regressor.staged_predict(X_test_reg)):
    test_errors.append(mean_absolute_error(y_test_reg, y_pred_stage))

ax3.plot(range(1, len(train_errors) + 1), train_errors, 'b-', label='Entrenamiento', alpha=0.7)
ax3.plot(range(1, len(test_errors) + 1), test_errors, 'r-', label='Prueba', alpha=0.7)
ax3.set_xlabel('Número de Árboles')
ax3.set_ylabel('Error Absoluto Medio (MAE)')
ax3.set_title('Evolución del Error durante Entrenamiento')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Distribución de residuos
residuos = y_test_reg - y_pred_reg
ax4.hist(residuos, bins=30, alpha=0.7, color='orange', edgecolor='black')
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Residuos (Error de Predicción)')
ax4.set_ylabel('Frecuencia')
ax4.set_title('Distribución de Residuos')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# RECOMENDACIONES CON RENDIMIENTO ESPPERADO
print("\n" + "="*60)
print("RECOMENDACIONES CON RENDIMIENTO ESPERADO")
print("="*60)

# Analizar mejores lotes para cada cultivo
cultivos_analizar = ['Soja', 'Maíz', 'Trigo', 'Sorgo', 'Algodón']

for cultivo in cultivos_analizar:
    # Filtrar lotes donde este cultivo es óptimo
    lotes_cultivo = df_reg[df_reg['Cultivo_Optimo'] == cultivo].copy() # Use .copy()

    if len(lotes_cultivo) > 0:
        mejor_lote_idx = lotes_cultivo['Rendimiento_Esperado'].idxmax()
        mejor_lote = df_reg.loc[mejor_lote_idx]

        print(f"\n MEJOR LOTE PARA {cultivo.upper()}:")
        print(f"   Rendimiento esperado: {mejor_lote['Rendimiento_Esperado']:,} kg/ha")
        print(f"   Características: pH={mejor_lote['pH']}, MO={mejor_lote['Materia_Organica']:.2f}%")
        print(f"   Nutrientes: N={mejor_lote['Nitrogeno']} ppm, P={mejor_lote['Fosforo']} ppm")
        print(f"   Textura: L={mejor_lote['Textura_Limoso']:.2f}, "
              f"A={mejor_lote['Textura_Arenoso']:.2f}, "
              f"Ar={mejor_lote['Textura_Arcilloso']:.2f}")

# PREDICCIÓN PARA NUEVOS LOTES
print("\n" + "="*60)
print("PREDICCIÓN PARA NUEVOS LOTES")
print("="*60)

nuevos_lotes = pd.DataFrame([
    [6.8, 3.2, 75, 35, 180, 0.5, 0.3, 0.2, 0],  # Soja
    [6.2, 3.8, 85, 45, 220, 0.4, 0.2, 0.4, 1],  # Maíz
    [7.1, 2.8, 90, 25, 150, 0.2, 0.3, 0.5, 2],  # Trigo
], columns=features + ['Cultivo_Cod'])

for i, (idx, lote) in enumerate(nuevos_lotes.iterrows()):
    # Convert lote Series to DataFrame with feature names
    lote_df = pd.DataFrame([lote], columns=features + ['Cultivo_Cod'])
    rendimiento_pred = gb_regressor.predict(lote_df)[0]
    cultivo_nombre = le.inverse_transform([int(lote['Cultivo_Cod'])])[0]

    print(f"\nLote Nuevo {i+1} ({cultivo_nombre}):")
    print(f"   Rendimiento predicho: {rendimiento_pred:,.0f} kg/ha")
    print(f"   Características: pH={lote['pH']}, MO={lote['Materia_Organica']}%")
    
    
