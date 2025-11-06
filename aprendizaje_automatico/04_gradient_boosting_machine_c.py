import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Boosting - Classifier
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
    
    
    # OUTPUT:
    # === GRADIENT BOOSTING CLASSIFIER - PREDICCIÓN DE CULTIVOS ===
# Total de lotes: 2000

# Distribución de cultivos óptimos:
# Cultivo_Optimo
# Sorgo      505
# Maíz       451
# Soja       441
# Trigo      382
# Algodón    221
# Name: count, dtype: int64

# Precisión del modelo: 0.442

# Reporte de Clasificación:
              # precision    recall  f1-score   support

     # Algodón       0.19      0.08      0.12        72
        # Maíz       0.40      0.48      0.44       139
        # Soja       0.38      0.40      0.39       129
       # Sorgo       0.62      0.67      0.65       141
       # Trigo       0.40      0.38      0.39       119

    # accuracy                           0.44       600
   # macro avg       0.40      0.40      0.40       600
# weighted avg       0.42      0.44      0.43       600

################################################

# ============================================================
# RECOMENDACIONES DE CULTIVOS PARA LOTES DE PRUEBA
# ============================================================

# Lote 1860:
  # Características: pH=5.7, MO=1.63%, N=28.0 ppm, P=45.0 ppm
  # Textura: Limoso=0.45, Arenoso=0.20, Arcilloso=0.34
  # Cultivo Real: Soja
  # Cultivo Recomendado: Maíz (confianza: 0.78)

# Lote 353:
  # Características: pH=8.3, MO=3.08%, N=82.0 ppm, P=69.0 ppm
  # Textura: Limoso=0.25, Arenoso=0.41, Arcilloso=0.33
  # Cultivo Real: Soja
  # Cultivo Recomendado: Sorgo (confianza: 0.64)

# Lote 1333:
  # Características: pH=8.3, MO=5.29%, N=105.0 ppm, P=32.0 ppm
  # Textura: Limoso=0.29, Arenoso=0.32, Arcilloso=0.39
  # Cultivo Real: Trigo
  # Cultivo Recomendado: Sorgo (confianza: 0.52)

# Lote 905:
  # Características: pH=7.9, MO=3.11%, N=107.0 ppm, P=74.0 ppm
  # Textura: Limoso=0.55, Arenoso=0.14, Arcilloso=0.31
  # Cultivo Real: Maíz
  # Cultivo Recomendado: Algodón (confianza: 0.45)
