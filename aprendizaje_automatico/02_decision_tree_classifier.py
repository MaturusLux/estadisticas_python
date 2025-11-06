# Preparación de los Datos
# Dados 20 lotes, y datos de entrada:

# Característica	Unidad	Tipo de Dato	Rol
# pH			(Numérico)	Variable de entrada (X)	
# Materia Orgánica	(%)	Variable de entrada (X)	
# Nitrógeno (N)		(ppm)	Variable de entrada (X)	
# Fósforo (P)		(ppm)	Variable de entrada (X)	
# Textura del Suelo	(Arcilloso/Limoso/Arenoso)	Variable de entrada (X)	
# Cultivo Óptimo		(Maíz/Frijol/Trigo/Pastos)	Variable Objetivo (y)	

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generación de datos simulados para 20 lotes (Reemplazar con tus datos reales)
np.random.seed(42)
data = {
    'pH': np.round(np.random.uniform(5.5, 7.5, 20), 1),
    'Materia_Organica': np.round(np.random.uniform(1.0, 5.0, 20), 2),
    'Nitrogeno': np.round(np.random.uniform(15, 60, 20), 0),
    'Textura': np.random.choice(['Arcilloso', 'Limoso', 'Arenoso'], 20),
    # Variable objetivo (Uso más apropiado: Clasificación)
    'Cultivo_Optimo': np.random.choice(['Maíz', 'Frijol', 'Trigo', 'Pastos'], 20, p=[0.3, 0.3, 0.2, 0.2])
}
df = pd.DataFrame(data)

# 1. Preparación: Codificación One-Hot para variables categóricas (Textura)
df_encoded = pd.get_dummies(df, columns=['Textura'], drop_first=True)

X = df_encoded.drop('Cultivo_Optimo', axis=1)
y = df_encoded['Cultivo_Optimo']

# 2. División de datos (Aunque con 20 es limitado, es buena práctica)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Características de entrada (X):\n", X_train.columns.tolist())
############################################
# Entrenamiento y Visualización del Árbol
# 3. Entrenamiento del modelo CART
# Parámetros: max_depth limita la complejidad, criterion='gini' es el estándar CART
modelo_arbol = DecisionTreeClassifier(max_depth=4, criterion='gini', random_state=42)
modelo_arbol.fit(X_train, y_train)

# 4. Evaluación (en este caso, la precisión de clasificación)
score = modelo_arbol.score(X_test, y_test)
print(f"\nPrecisión del Modelo (Clasificación): {score:.2f}")

# 5. Visualización del Árbol (para interpretar las reglas)
plt.figure(figsize=(20, 10))
plot_tree(modelo_arbol, 
          feature_names=X.columns.tolist(),  
          class_names=modelo_arbol.classes_,
          filled=True, 
          rounded=True,
          fontsize=10)
plt.title("Árbol de Decisión para Determinar Cultivo Óptimo (CART)")
plt.show()

###########################################
print("\nImportancia de las Características:")
for name, importance in zip(X.columns, modelo_arbol.feature_importances_):
    print(f"  {name}: {importance:.3f}")

# Ejemplo de predicción para un nuevo lote:
nuevo_lote = pd.DataFrame({
    'pH': [6.0], 
    'Materia_Organica': [4.5], 
    'Nitrogeno': [55.0], 
    'Textura_Limoso': [1], 
    'Textura_Arenoso': [0]
})

prediccion = modelo_arbol.predict(nuevo_lote)
print(f"\nEl cultivo óptimo predicho para el nuevo lote es: **{prediccion[0]}**")
