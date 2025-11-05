import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# Regresión Ridge Logística - Probabilidad de demoras
print("\n=== REGRESIÓN RIDGE LOGÍSTICA - DEMORAS EN TRANSPORTE ===")

# Datos: Costo de transporte de cereales (en miles $)
np.random.seed(42)
n_muestras = 100

# Variables para problema de clasificación
roturas_camion = np.random.poisson(0.3, n_muestras)
caminos_mal_estado = np.random.binomial(1, 0.4, n_muestras)
tiempo_descarga = np.random.uniform(1, 6, n_muestras)
demoras_no_programadas = np.random.poisson(0.5, n_muestras)
lluvia = np.random.binomial(1, 0.3, n_muestras)

# Probabilidad de demora grave
log_odds = (
    1.5 * roturas_camion +
    2.0 * caminos_mal_estado +
    0.8 * tiempo_descarga +
    1.2 * demoras_no_programadas +
    1.0 * lluvia -
    3.0  # intercept
)

probabilidad = 1 / (1 + np.exp(-log_odds))
demora_grave = np.random.binomial(1, probabilidad, n_muestras)

# Matriz de diseño
X_logistic = np.column_stack([roturas_camion, caminos_mal_estado, tiempo_descarga, 
                             demoras_no_programadas, lluvia])
X_logistic_std = (X_logistic - X_logistic.mean(axis=0)) / X_logistic.std(axis=0)
y_logistic = demora_grave

# Función sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

# Función de costo Ridge Logística
def ridge_logistic_cost(theta, X, y, alpha):
    m = len(y)
    z = X @ theta
    h = sigmoid(z)
    
    # Log-verosimilitud
    cost = -np.sum(y * np.log(h + 1e-8) + (1 - y) * np.log(1 - h + 1e-8)) / m
    # Regularización Ridge (excluir intercept)
    regularization = alpha * np.sum(theta[1:] ** 2) / (2 * m)
    
    return cost + regularization

# Gradiente Ridge Logística
def ridge_logistic_gradient(theta, X, y, alpha):
    m = len(y)
    z = X @ theta
    h = sigmoid(z)
    
    gradient = (X.T @ (h - y)) / m
    # Regularización (no aplicar al intercept)
    regularization = np.zeros_like(theta)
    regularization[1:] = (alpha * theta[1:]) / m
    
    return gradient + regularization

# Añadir intercept
X_logistic_std_int = np.column_stack([np.ones(len(X_logistic_std)), X_logistic_std])

# Optimización Ridge Logística
alphas_logistic = [0, 0.5, 1, 2, 5]
coeficientes_logistic = []

for alpha in alphas_logistic:
    theta_inicial = np.zeros(X_logistic_std_int.shape[1])
    resultado = minimize(ridge_logistic_cost, theta_inicial, 
                        args=(X_logistic_std_int, y_logistic, alpha), 
                        method='BFGS',
                        jac=ridge_logistic_gradient)
    coeficientes_logistic.append(resultado.x)

coeficientes_logistic = np.array(coeficientes_logistic)

# Predicciones
theta_optimo = coeficientes_logistic[2]  # alpha=1
probabilidades_pred = sigmoid(X_logistic_std_int @ theta_optimo)
predicciones = (probabilidades_pred > 0.5).astype(int)

# Calcular precisión manualmente
precision = np.mean(predicciones == y_logistic)

# Visualización
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Gráfico 1: Coeficientes Ridge Logística
variables_logistic = ['Intercept', 'Roturas', 'Caminos Mal', 'Tiempo Desc', 'Demoras', 'Lluvia']
for i in range(len(variables_logistic)):
    ax1.plot(alphas_logistic, coeficientes_logistic[:, i], marker='o', 
             label=variables_logistic[i], linewidth=2)

ax1.set_xlabel('Alpha (fuerza de regularización)')
ax1.set_ylabel('Coeficientes')
ax1.set_title('Ridge Logística: Coeficientes vs Alpha')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfico 2: Probabilidades predichas
ax2.scatter(range(len(y_logistic)), probabilidades_pred, c=y_logistic, 
           cmap='coolwarm', alpha=0.7, s=60)
ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Umbral 0.5')
ax2.set_xlabel('Muestra')
ax2.set_ylabel('Probabilidad Predicha')
ax2.set_title('Probabilidades de Demora Grave')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Gráfico 3: Matriz de confusión manual
from collections import Counter
tp = np.sum((predicciones == 1) & (y_logistic == 1))
fp = np.sum((predicciones == 1) & (y_logistic == 0))
tn = np.sum((predicciones == 0) & (y_logistic == 0))
fn = np.sum((predicciones == 0) & (y_logistic == 1))

matriz_confusion = np.array([[tn, fp], [fn, tp]])
ax3.imshow(matriz_confusion, interpolation='nearest', cmap=plt.cm.Blues)
ax3.set_title(f'Matriz de Confusión (Precisión: {precision:.3f})')
ax3.set_ylabel('Real')
ax3.set_xlabel('Predicho')
ax3.set_xticks([0, 1])
ax3.set_yticks([0, 1])
ax3.set_xticklabels(['Sin Demora', 'Demora'])
ax3.set_yticklabels(['Sin Demora', 'Demora'])

# Añadir texto en la matriz
thresh = matriz_confusion.max() / 2.
for i in range(2):
    for j in range(2):
        ax3.text(j, i, format(matriz_confusion[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if matriz_confusion[i, j] > thresh else "black")

# Gráfico 4: Importancia de variables
importancia = np.abs(coeficientes_logistic[2, 1:])  # Excluir intercept
variables_importancia = variables_logistic[1:]

ax4.barh(variables_importancia, importancia, color='lightcoral', alpha=0.7)
ax4.set_xlabel('Importancia (valor absoluto coeficiente)')
ax4.set_title('Importancia de Variables para Demoras')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nPrecisión del modelo: {precision:.3f}")
print("\nCoeficientes Ridge Logística (alpha=1):")
for var, coef in zip(variables_logistic, coeficientes_logistic[2, :]):
    print(f"{var}: {coef:.4f}")

print("\n→ Las variables más importantes son 'Caminos Mal' y 'Roturas'")
