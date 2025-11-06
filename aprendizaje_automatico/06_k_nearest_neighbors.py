import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split # Añadimos split para una evaluación más realista
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- 1. Generar Datos Simulados (Lotes de Campo) ---

np.random.seed(42)
N_LOTES = 300 # Más lotes para mejor entrenamiento

# Características del Lote que influencian la elección de Herbicida:
# C1 - Densidad_Maleza_M1: Intensidad de la maleza dominante 1 (0-10)
# C2 - Etapa_Crecimiento_M2: Etapa de crecimiento de la maleza dominante 2 (0-10, 10=madura)
# C3 - pH_Suelo: Nivel de pH (5.5 - 8.5)

# Simulamos 2 grupos (clases) que responden mejor a un herbicida específico:

# Grupo H_A (Herbicida A, ej. Glifosato): Mejor para malezas densas y pH bajo
grupo_ha = np.random.randn(150, 3) * np.array([2, 1, 0.5]) + np.array([8, 3, 6.0])

# Grupo H_B (Herbicida B, ej. 2,4-D): Mejor para malezas en etapa tardía y pH alto
grupo_hb = np.random.randn(150, 3) * np.array([1, 2, 0.5]) + np.array([3, 7, 7.5])

X = np.vstack([grupo_ha, grupo_hb])
# Clases: 0 = Herbicida A, 1 = Herbicida B
y = np.hstack([np.zeros(150), np.ones(150)])

# Asegurar que las características estén en un rango razonable
X[:, 0] = np.clip(X[:, 0], 0, 10)
X[:, 1] = np.clip(X[:, 1], 0, 10)
X[:, 2] = np.clip(X[:, 2], 5.5, 8.5)

feature_names = ['Densidad_Maleza_M1', 'Etapa_Crecimiento_M2', 'pH_Suelo']
X_df = pd.DataFrame(X, columns=feature_names)
X_df['Herbicida_Real'] = y

print(f"Datos generados para {N_LOTES} lotes de campo.")
print(X_df.head())

# --- 2. Estandarización de Datos (Crucial para k-NN) ---

# k-NN se basa en la distancia, por lo que las variables deben tener una escala similar.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. División y Entrenamiento del Clasificador k-NN ---

# Dividimos en entrenamiento y prueba para evaluar el rendimiento
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# k = 5 vecinos es un valor común y par (mejor impar para 2 clases)
k = 5
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Entrenamos el modelo con los datos de entrenamiento
knn_classifier.fit(X_train, y_train)

# Predecimos en el conjunto de prueba
predicciones_herbicida = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, predicciones_herbicida)

print(f"\nPrecisión del modelo k-NN (k={k}) en el set de prueba: {accuracy:.3f}")


# --- 4. Sistema de Recomendación de Herbicidas ---

# Definimos las recomendaciones finales
recomendaciones_herbicida = {
    0.0: "Recomendación: Herbicida A (Glifosato, ej. para alta Densidad y bajo pH)",
    1.0: "Recomendación: Herbicida B (2,4-D, ej. para Etapa Tardía y alto pH)"
}

# Función para recomendar a un nuevo lote
def recomendar_herbicida_nuevo_lote(datos_lote, scaler_obj, knn_model, recomendaciones_dict):
    """
    Clasifica un nuevo lote y devuelve la recomendación.
    datos_lote debe ser un array/lista de [Densidad_Maleza_M1, Etapa_Crecimiento_M2, pH_Suelo]
    """
    # Escalar el nuevo dato con el mismo scaler usado para entrenar
    datos_lote_scaled = scaler_obj.transform(np.array(datos_lote).reshape(1, -1))
    
    # Predecir la clase (herbicida)
    herbicida_predicho = knn_model.predict(datos_lote_scaled)[0]
    
    return recomendaciones_dict[herbicida_predicho], int(herbicida_predicho)

# Ejemplo de uso: Nuevo Lote
# Lote con alta densidad (8), etapa temprana (2), pH bajo (6.2) -> Debería ser Herbicida A (0)
nuevo_lote_data = [8.5, 2.5, 6.2] 
rec, clase = recomendar_herbicida_nuevo_lote(
    nuevo_lote_data, scaler, knn_classifier, recomendaciones_herbicida
)
print(f"\nNuevo Lote (Densidad={nuevo_lote_data[0]}, Etapa={nuevo_lote_data[1]}, pH={nuevo_lote_data[2]}):")
print(f"  Clasificado como Herbicida {'A' if clase == 0 else 'B'}")
print(f"  {rec}")


# --- 5. Visualización con Matplotlib (Densidad M1 vs Etapa M2) ---

plt.figure(figsize=(10, 6))
cmap_bold = ListedColormap(['#1f77b4', '#ff7f0e']) # Azul para Herbicida A, Naranja para Herbicida B

scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=40, alpha=0.7)
plt.scatter(scaler.transform(np.array(nuevo_lote_data).reshape(1, -1))[:, 0], 
            scaler.transform(np.array(nuevo_lote_data).reshape(1, -1))[:, 1], 
            c='red', marker='X', s=200, label='Nuevo Lote (Predicción)', zorder=3)

plt.xlabel(f'{feature_names[0]} (Estandarizado)')
plt.ylabel(f'{feature_names[1]} (Estandarizado)')
plt.title(f'Clasificación k-NN de Herbicidas ({feature_names[0]} vs {feature_names[1]})')

# Crear leyenda
legend = plt.legend(
    handles=scatter.legend_elements()[0] + [plt.scatter([], [], c='red', marker='X', s=200)], 
    labels=['Herbicida A', 'Herbicida B', 'Nuevo Lote'], 
    title="Clase"
)
plt.grid(True, alpha=0.3)
plt.show()

