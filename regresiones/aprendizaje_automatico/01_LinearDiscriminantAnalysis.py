import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.stats import multivariate_normal

# Datos: Monitoreo de enfermedades en cultivo de maíz
np.random.seed(42)
n_muestras = 300

# Generar datos para 3 enfermedades comunes en maíz
enfermedades = {
    0: "Roya Común (Fungo)",
    1: "Tizón Foliar (Bacteria)", 
    2: "Fusarium (Mazorca)"
}

# Parámetros para cada enfermedad - con más variabilidad
params = {
    0: {  # Roya Común
        'temperatura': (18, 25),
        'humedad': (70, 90), 
        'lesion_follaje': (15, 60),
        'lesion_tallo': (1, 4),
        'presencia_esporas': (3, 5),
        'color_lesion': (0.8, 1.2)  # Rango en lugar de valor fijo
    },
    1: {  # Tizón Foliar
        'temperatura': (25, 32),
        'humedad': (80, 95),
        'lesion_follaje': (20, 70),
        'lesion_tallo': (2, 6),
        'presencia_esporas': (0, 2),
        'color_lesion': (1.8, 2.2)
    },
    2: {  # Fusarium
        'temperatura': (20, 30),
        'humedad': (60, 85),
        'lesion_follaje': (5, 30),
        'lesion_tallo': (4, 8),
        'presencia_esporas': (2, 4),
        'color_lesion': (2.8, 3.2)
    }
}

# Generar datos sintéticos con más variabilidad
X = []
y = []

for enfermedad_id, n in enumerate([100, 100, 100]):
    params_enfer = params[enfermedad_id]
    
    for _ in range(n):
        muestra = [
            np.random.uniform(*params_enfer['temperatura']) + np.random.normal(0, 1.5),
            np.random.uniform(*params_enfer['humedad']) + np.random.normal(0, 4),
            max(1, np.random.uniform(*params_enfer['lesion_follaje']) + np.random.normal(0, 8)),
            max(0.5, np.random.uniform(*params_enfer['lesion_tallo']) + np.random.normal(0, 1)),
            max(0, np.random.uniform(*params_enfer['presencia_esporas']) + np.random.normal(0, 0.5)),
            np.random.uniform(*params_enfer['color_lesion']) + np.random.normal(0, 0.2)
        ]
        
        X.append(muestra)
        y.append(enfermedad_id)

X = np.array(X)
y = np.array(y)

print("=== LDA - CLASIFICACIÓN DE ENFERMEDADES EN MAÍZ ===")
print(f"Muestras totales: {len(X)}")
print(f"Distribución por enfermedad:")
for enf_id, enf_nombre in enfermedades.items():
    count = np.sum(y == enf_id)
    print(f"  {enf_nombre}: {count} muestras")

# LDA CORREGIDO - CON MANEJO DE SINGULARIDAD
class LDA:
    def __init__(self, regularization=1e-6):
        self.regularization = regularization
        self.means_ = None
        self.covariance_ = None
        self.priors_ = None
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Calcular priors
        self.priors_ = np.array([np.mean(y == c) for c in self.classes_])
        
        # Calcular medias por clase
        self.means_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        
        # Calcular matriz de covarianza común CON REGULARIZACIÓN
        self.covariance_ = np.zeros((n_features, n_features))
        for c in self.classes_:
            X_c = X[y == c]
            self.covariance_ += (X_c - self.means_[c]).T @ (X_c - self.means_[c])
        self.covariance_ /= (n_samples - n_classes)
        
        # Añadir regularización para evitar singularidad
        self.covariance_ += np.eye(n_features) * self.regularization
        
        # Verificar que la matriz sea invertible
        try:
            # Calcular coeficientes para la función discriminante
            self.coef_ = linalg.solve(self.covariance_, self.means_.T, assume_a='sym').T
        except linalg.LinAlgError:
            # Si falla, usar pseudoinversa
            print("Usando pseudoinversa debido a problemas de singularidad...")
            self.coef_ = (self.means_ @ linalg.pinv(self.covariance_)).T
        
        self.intercept_ = -0.5 * np.diag(self.means_ @ self.coef_.T) + np.log(self.priors_)
        
        return self
    
    def predict(self, X):
        decision_scores = X @ self.coef_.T + self.intercept_
        return self.classes_[np.argmax(decision_scores, axis=1)]
    
    def predict_proba(self, X):
        decision_scores = X @ self.coef_.T + self.intercept_
        exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

# Aplicar LDA
print("\n=== ENTRENANDO MODELO LDA ===")
lda = LDA(regularization=1e-4)  # Regularización aumentada
lda.fit(X, y)
y_pred = lda.predict(X)

# Calcular precisión
accuracy = np.mean(y_pred == y)
print(f"Precisión del modelo: {accuracy:.3f}")

# Matriz de confusión
print("\nMatriz de Confusión:")
conf_matrix = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        conf_matrix[i, j] = np.sum((y == i) & (y_pred == j))

print("Real \\ Predicho\tRoya\tTizón\tFusarium")
for i, enf_real in enumerate(["Roya", "Tizón", "Fusarium"]):
    print(f"{enf_real:<15}", end="")
    for j in range(3):
        print(f"{int(conf_matrix[i, j]):<8}", end="")
    print()

# VISUALIZACIÓN EN 2D
# def lda_transform(X, lda_model, n_components=2):
#     """Transformar datos al espacio discriminante"""
#     return X @ lda_model.coef_[:, :n_components].T
def lda_transform(X, lda_model, n_components=2):
    """Proyección lineal en el espacio discriminante (usa todas las features)"""
    scores = X @ lda_model.coef_.T  # (n_samples, n_classes)
    return scores[:, :n_components]

X_lda = lda_transform(X, lda)

# VISUALIZACIONES
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Gráfico 1: Proyección LDA
colors = ['red', 'blue', 'green']
markers = ['o', 's', '^']

for i, enf_nombre in enfermedades.items():
    mask = y == i
    ax1.scatter(X_lda[mask, 0], X_lda[mask, 1], 
               c=colors[i], marker=markers[i], 
               label=enf_nombre, alpha=0.7, s=60)

ax1.set_xlabel('Función Discriminante 1')
ax1.set_ylabel('Función Discriminante 2')
ax1.set_title('Proyección LDA - Separación de Enfermedades')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfico 2: Importancia de variables en LDA
variables = ['Temperatura', 'Humedad', 'Lesión Follaje', 'Lesión Tallo', 'Esporas', 'Color']
importancia = np.abs(lda.coef_).mean(axis=0)

ax2.barh(variables, importancia, color='lightcoral', alpha=0.7)
ax2.set_xlabel('Importancia (valor absoluto coeficientes LDA)')
ax2.set_title('Importancia de Variables en la Clasificación')
ax2.grid(True, alpha=0.3)

# Gráfico 3: Probabilidades por clase (primeras 50 muestras)
probas = lda.predict_proba(X)
muestras_a_mostrar = 50
x_pos = np.arange(muestras_a_mostrar)
ancho = 0.25

for i, enf_nombre in enumerate(enfermedades.values()):
    ax3.bar(x_pos + i*ancho, probas[:muestras_a_mostrar, i], ancho, label=enf_nombre, alpha=0.7)

ax3.set_xlabel('Muestras')
ax3.set_ylabel('Probabilidad')
ax3.set_title('Probabilidades de Clasificación (Primeras 50 Muestras)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Gráfico 4: Características por enfermedad
caracteristicas_plot = ['Temperatura', 'Humedad', 'Lesión Follaje']
x_pos_caract = np.arange(len(caracteristicas_plot))

for i, enf_nombre in enumerate(enfermedades.values()):
    medias = [np.mean(X[y == i, j]) for j in range(3)]
    ax4.bar(x_pos_caract + i*0.25, medias, 0.25, label=enf_nombre, alpha=0.7)

ax4.set_xlabel('Características')
ax4.set_ylabel('Valor Promedio')
ax4.set_title('Perfil Promedio por Enfermedad')
ax4.set_xticks(x_pos_caract + 0.25)
ax4.set_xticklabels(caracteristicas_plot)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# RECOMENDACIONES DE AGROQUÍMICOS
print("\n" + "="*70)
print("RECOMENDACIONES DE APLICACIÓN DE AGROQUÍMICOS")
print("="*70)

recomendaciones = {
    0: {  # Roya Común
        "nombre": "ROYA COMÚN (Puccinia sorghi)",
        "sintomas_principales": "Pústulas pequeñas color naranja-amarillo en ambas caras de hojas",
        "condiciones_favorables": "Temperaturas frescas (18-25°C) y alta humedad relativa",
        "agroquimicos_recomendados": [
            "Triazoles: Tebuconazol 25% (0.5-0.8 L/ha)",
            "Estrobilurinas: Azoxystrobin 23% (0.3-0.5 L/ha)", 
            "Mezcla: Pyraclostrobin + Epoxiconazol (0.4 L/ha)"
        ],
        "momento_aplicacion": "Al primer síntoma, repetir cada 15 días si condiciones persisten",
        "observaciones": "Aplicar con cobertura completa del follaje. Rotar modos de acción."
    },
    
    1: {  # Tizón Foliar
        "nombre": "TIZÓN FOLIAR (Exserohilum turcicum)",
        "sintomas_principales": "Lesiones alargadas color café oscuro, forma rectangular en hojas",
        "condiciones_favorables": "Temperaturas cálidas (25-32°C) con rocío prolongado",
        "agroquimicos_recomendados": [
            "Clorotalonil 72% (1.5-2.0 L/ha)",
            "Mancozeb 75% (1.8-2.5 kg/ha)",
            "Mefenoxam + Clorotalonil (1.2 L/ha)"
        ],
        "momento_aplicacion": "Preventivo en V8 o al primer síntoma. Intervalo 10-12 días",
        "observaciones": "Aplicar con alto volumen de agua (200-300 L/ha). Cubrir bien el tercio medio."
    },
    
    2: {  # Fusarium
        "nombre": "FUSARIUM (Fusarium verticillioides)",
        "sintomas_principales": "Podredumbre rosada en mazorca, estrías en tallo, marchitez",
        "condiciones_favorables": "Estrés hídrico seguido de humedad en floración",
        "agroquimicos_recomendados": [
            "Tiofanato metílico 70% (1.0-1.5 kg/ha)",
            "Carbendazim 50% (0.8-1.2 kg/ha)", 
            "Protioconazol 25% (0.3-0.5 L/ha)"
        ],
        "momento_aplicacion": "Aplicar en floración y llenado de grano. 2-3 aplicaciones",
        "observaciones": "Combinar con control de insectos barrenador. Tratamiento de semilla recomendado."
    }
}

# MOSTRAR RECOMENDACIONES ESPECÍFICAS
print("\n🔍 BASADO EN LA CLASIFICACIÓN LDA, SE RECOMIENDA:")

for enf_id in range(3):
    count_pred = np.sum(y_pred == enf_id)
    if count_pred > 0:
        reco = recomendaciones[enf_id]
        print(f"\n🎯 **{reco['nombre']}** - {count_pred} lotes afectados")
        print(f"📋 Síntomas: {reco['sintomas_principales']}")
        print(f"🌡️ Condiciones: {reco['condiciones_favorables']}")
        print("💊 FORMULAS RECOMENDADAS:")
        for formula in reco['agroquimicos_recomendados']:
            print(f"   • {formula}")
        print(f"⏰ Momento: {reco['momento_aplicacion']}")
        print(f"💡 Observaciones: {reco['observaciones']}")

# ANÁLISIS DE LAS VARIABLES MÁS IMPORTANTES
print("\n" + "="*70)
print("ANÁLISIS DE VARIABLES DISCRIMINANTES")
print("="*70)

print("Variables ordenadas por importancia:")
indices_importancia = np.argsort(importancia)[::-1]
for idx in indices_importancia:
    print(f"  {variables[idx]}: {importancia[idx]:.4f}")

# SIMULACIÓN DE NUEVOS CASOS
print("\n" + "="*70)
print("SIMULACIÓN - DIAGNÓSTICO DE NUEVOS CASOS")
print("="*70)

# Casos de prueba
nuevos_casos = [
    [22, 85, 45, 3, 4, 1.1],   # Probable Roya
    [28, 90, 60, 4, 1, 2.0],   # Probable Tizón
    [25, 75, 20, 6, 3, 3.0],   # Probable Fusarium
    [20, 80, 35, 5, 2, 2.5]    # Caso intermedio
]

descripciones_casos = [
    "Temperatura fresca, alta humedad, esporas naranjas",
    "Temperatura cálida, muy húmedo, lesiones oscuras", 
    "Temperatura media, lesiones en tallo, mazorca afectada",
    "Síntomas mixtos - diagnóstico complejo"
]

for i, (caso, desc) in enumerate(zip(nuevos_casos, descripciones_casos)):
    caso_array = np.array(caso).reshape(1, -1)
    prediccion = lda.predict(caso_array)[0]
    probabilidades = lda.predict_proba(caso_array)[0]
    
    print(f"\nCaso {i+1}: {desc}")
    print(f"Diagnóstico: {enfermedades[prediccion]}")
    print("Probabilidades:")
    for enf_id, prob in enumerate(probabilidades):
        print(f"  {enfermedades[enf_id]}: {prob:.3f}")
