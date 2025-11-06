from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# --- 1. Datos Simulados de Conteo de Síntomas ---
# X: [Manchas_Foliar, Daño_Tallo, Conteo_Esporas]
# Y: [0=Fúngica, 1=Bacteriana, 2=Viral]
np.random.seed(42)

# Fúngica (Alta mancha, Baja espora, Bajo tallo)
X_fung = np.random.randint(5, 15, size=(50, 1)) * np.array([1, 0.1, 0.5])
X_fung = X_fung.astype(int)
y_fung = np.zeros(50)

# Bacteriana (Bajo mancha, Alto tallo, Sin espora)
X_bact = np.random.randint(2, 10, size=(50, 1)) * np.array([0.5, 1, 0.01])
X_bact = X_bact.astype(int)
y_bact = np.ones(50)

# Viral (Manchas moderadas, Tallo moderado, Alta espora)
X_vir = np.random.randint(5, 10, size=(50, 1)) * np.array([1, 1, 1]) + np.array([0, 0, 5])
X_vir = X_vir.astype(int)
y_vir = np.full(50, 2)

X = np.vstack([X_fung, X_bact, X_vir])
y = np.hstack([y_fung, y_bact, y_vir])

# --- 2. Preparación y Entrenamiento ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Multinomial Naive Bayes es ideal para conteo (valores discretos no negativos)
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# Predicción y Reporte
y_pred = mnb.predict(X_test)

print("=== Resultados del Naive Bayes Multinomial ===")
print("Clases (0=Fúngica, 1=Bacteriana, 2=Viral)")
print(classification_report(y_test, y_pred))

# --- 3. Uso en Diagnóstico ---

# Nuevo caso: Alta mancha (12), Daño de tallo bajo (2), Sin esporas (0)
nuevo_sintoma = np.array([[12, 2, 0]]) 

probas = mnb.predict_proba(nuevo_sintoma)[0]
clase_pred = mnb.predict(nuevo_sintoma)[0]

clases_map = {0: "Fúngica", 1: "Bacteriana", 2: "Viral"}

print(f"\nSíntomas (Mancha/Tallo/Esporas): {nuevo_sintoma[0]}")
print(f"Diagnóstico Predicho: **{clases_map[clase_pred]}**")
print(f"Probabilidades por Clase:")
for i, p in enumerate(probas):
    print(f"  {clases_map[i]}: {p:.3f}")


#   Viral: 0.001
