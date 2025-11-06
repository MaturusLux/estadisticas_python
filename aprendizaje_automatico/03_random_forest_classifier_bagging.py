import pandas as pd
from sklearn.model_selection import train_test_split
# Bagging Ensemble (cada árbol es entrenado independientemente)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Generación de Datos Simulados (2000-2024, 5 Cultivos) ---
np.random.seed(42)
num_lotes = 2000
data = {
    'pH': np.round(np.random.uniform(5.0, 8.0, num_lotes), 1),
    'Materia_Organica': np.round(np.random.uniform(0.5, 6.0, num_lotes), 2),
    'Nitrogeno': np.round(np.random.uniform(10, 80, num_lotes), 0),
    'Fosforo': np.round(np.random.uniform(10, 50, num_lotes), 0),
    'Textura': np.random.choice(['Arcilloso', 'Limoso', 'Arenoso'], num_lotes)
}
# Variable Objetivo: Clasificación entre 5 tipos de cultivos
cultivos = ['Soja', 'Trigo', 'Sorgo', 'Algodon', 'Girasol']
data['Cultivo_Optimo'] = np.random.choice(cultivos, num_lotes, p=[0.25, 0.2, 0.2, 0.15, 0.2])

df = pd.DataFrame(data)

# Codificación One-Hot para variables categóricas
df_encoded = pd.get_dummies(df, columns=['Textura'], drop_first=True)
X = df_encoded.drop('Cultivo_Optimo', axis=1)
y = df_encoded['Cultivo_Optimo']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Entrenamiento del Random Forest ---
# n_estimators=200: Usamos 200 árboles para mejorar la precisión y la robustez.
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Evaluación
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Precisión del Random Forest con 200 árboles: {rf_accuracy:.4f}")

# Precisión del Random Forest con 200 árboles: 0.2100

######################################
# Obtener la importancia de las características
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Imprimir la importancia
print("\n--- Importancia de Características (Random Forest) ---")
print(feature_importances)

# Visualización (requiere Matplotlib)
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='barh')
plt.title("Importancia de Características para la Selección de Cultivo")
plt.xlabel("Nivel de Importancia (basado en reducción de impureza)")
plt.ylabel("Características del Suelo")
plt.gca().invert_yaxis()
plt.show()

# --- Importancia de Características (Random Forest) ---
# Materia_Organica    0.270904
# Nitrogeno           0.238069
# Fosforo             0.219244
# pH                  0.201192
# Textura_Limoso      0.036632
# Textura_Arenoso     0.033959
# dtype: float64
##########################################
