import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

# Configuración de Matplotlib
plt.style.use('seaborn-v0_8-whitegrid')

# --- 1. Generar Datos Simulados (3 Cultivos) ---
# Simulación de datos para 3 clases (Cultivo A, B, C)
np.random.seed(42)
N_MUESTRAS = 300

# Características (X): Precio_Venta_USD, Costo_Insumos_USD
# Suponemos que la "Clase" (Y) puede ser la calidad final o la zona de origen.

# Cultivo A (Clase 0): Precios Medios, Baja varianza.
X0 = np.random.randn(N_MUESTRAS // 3, 2) * 1.5 + np.array([100, 10])

# Cultivo B (Clase 1): Precios Altos, Alta varianza y dispersión inclinada.
X1 = np.random.randn(N_MUESTRAS // 3, 2) * np.array([4, 2]) + np.array([120, 15])
X1 = X1.dot([[1, 0.5], [0, 1]]) # Inclinamos el cluster (covarianza diferente)

# Cultivo C (Clase 2): Precios Bajos, Alta varianza.
X2 = np.random.randn(N_MUESTRAS - N_MUESTRAS // 3 * 2, 2) * 3 + np.array([90, 5])

X = np.vstack([X0, X1, X2])
y = np.hstack([np.zeros(len(X0)), np.ones(len(X1)), np.full(len(X2), 2)])

feature_names = ['Precio_Venta_USD', 'Costo_Insumos_USD']
class_names = ['Cultivo_A', 'Cultivo_B', 'Cultivo_C']

# --- 2. Preparación y Estandarización de Datos ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# La estandarización es crucial para modelos basados en distancia/varianza
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- 3. Entrenamiento de LDA y QDA ---

# LDA: Frontera Lineal (asume misma covarianza)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_scaled, y_train)

# QDA: Frontera Cuadrática (asume covarianzas diferentes)
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train_scaled, y_train)


# --- 4. Evaluación de Precisión (Accuracy) ---

lda_pred = lda.predict(X_test_scaled)
qda_pred = qda.predict(X_test_scaled)

lda_accuracy = accuracy_score(y_test, lda_pred)
qda_accuracy = accuracy_score(y_test, qda_pred)

print("=== Precisión de los Modelos ===")
print(f"Precisión LDA (Frontera Lineal): {lda_accuracy:.4f}")
print(f"Precisión QDA (Frontera Cuadrática): {qda_accuracy:.4f}")
# En este caso simulado, QDA debería tener una mejor precisión debido a la forma de los clusters.


# --- 5. Visualización de las Fronteras de Decisión ---

# Función para graficar la frontera de decisión
def plot_decision_boundary(model, X_data, y_data, title, ax):
    # Crear un meshgrid (rejilla de puntos) para clasificar toda la superficie
    x_min, x_max = X_data[:, 0].min() - 1, X_data[:, 0].max() + 1
    y_min, y_max = X_data[:, 1].min() - 1, X_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Predecir cada punto de la rejilla
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Definir mapas de colores
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # Graficar la superficie de predicción
    ax.pcolormesh(xx, yy, Z, cmap=cmap_light, alpha=0.5)

    # Graficar los puntos de datos de entrenamiento
    scatter = ax.scatter(X_data[:, 0], X_data[:, 1], c=y_data, cmap=cmap_bold,
                         edgecolor='k', s=20)
    
    # Etiquetas y título
    ax.set_title(title)
    ax.set_xlabel(feature_names[0] + ' (Escalado)')
    ax.set_ylabel(feature_names[1] + ' (Escalado)')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    
    # Leyenda
    ax.legend(handles=scatter.legend_elements()[0], labels=class_names, title="Cultivo Clase")


# Crear la figura con dos subgráficos
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Gráfico 1: LDA (Fronteras Lineales)
plot_decision_boundary(lda, X_train_scaled, y_train, 
                       f'LDA (Fronteras Lineales) - Precisión Test: {lda_accuracy:.4f}', axes[0])

# Gráfico 2: QDA (Fronteras Cuadráticas)
plot_decision_boundary(qda, X_train_scaled, y_train, 
                       f'QDA (Fronteras Cuadráticas) - Precisión Test: {qda_accuracy:.4f}', axes[1])

plt.tight_layout()
plt.show()

# === Precisión de los Modelos ===
# Precisión LDA (Frontera Lineal): 0.9889
# Precisión QDA (Frontera Cuadrática): 0.9889
