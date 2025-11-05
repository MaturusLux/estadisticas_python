import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Datos: Costo de transporte de cereales (en miles $)
# Variables: distancia_km, toneladas, combustible, peajes, mantenimiento
np.random.seed(42)
n_muestras = 100

# Variables predictoras
distancia_km = np.random.uniform(50, 500, n_muestras)
toneladas = np.random.uniform(10, 50, n_muestras)
combustible = np.random.uniform(200, 800, n_muestras)
peajes = np.random.uniform(50, 300, n_muestras)
mantenimiento = np.random.uniform(100, 500, n_muestras)

# Variable objetivo: costo_total (con algunas variables irrelevantes)
costo_total = (
    0.8 * distancia_km + 
    1.2 * toneladas + 
    0.01 * combustible + 
    0.0 * peajes +  # Variable irrelevante
    0.0 * mantenimiento +  # Variable irrelevante
    np.random.normal(0, 50, n_muestras)
)

print("=== REGRESIÓN LASSO - COSTO DE TRANSPORTE ===")
print("Variables: distancia_km, toneladas, combustible, peajes, mantenimiento")

# Estandarizar variables
X = np.column_stack([distancia_km, toneladas, combustible, peajes, mantenimiento])
X_std = (X - X.mean(axis=0)) / X.std(axis=0)
y_std = (costo_total - costo_total.mean()) / costo_total.std()

# Función de costo Lasso
def lasso_cost(theta, X, y, alpha):
    m = len(y)
    predictions = X @ theta
    mse = np.sum((predictions - y) ** 2) / (2 * m)
    regularization = alpha * np.sum(np.abs(theta))
    return mse + regularization

# Gradiente de Lasso
def lasso_gradient(theta, X, y, alpha):
    m = len(y)
    predictions = X @ theta
    gradient = (X.T @ (predictions - y)) / m
    # Subgradiente para L1 (manejar discontinuidad en 0)
    subgradient = alpha * np.sign(theta)
    subgradient[theta == 0] = alpha  # En 0, el subgradiente está en [-alpha, alpha]
    return gradient + subgradient

# Optimización para diferentes alphas
alphas = [0, 0.1, 0.5, 1, 2]
coeficientes_lasso = []

for alpha in alphas:
    theta_inicial = np.zeros(X_std.shape[1])
    resultado = minimize(lasso_cost, theta_inicial, 
                        args=(X_std, y_std, alpha), 
                        method='BFGS',
                        jac=lasso_gradient)
    coeficientes_lasso.append(resultado.x)

coeficientes_lasso = np.array(coeficientes_lasso)

# Visualización
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico 1: Coeficientes vs Alpha
variables = ['Distancia', 'Toneladas', 'Combustible', 'Peajes', 'Mantenimiento']
for i in range(len(variables)):
    ax1.plot(alphas, coeficientes_lasso[:, i], marker='o', label=variables[i], linewidth=2)

ax1.set_xlabel('Alpha (fuerza de regularización)')
ax1.set_ylabel('Coeficientes')
ax1.set_title('Regularización Lasso: Coeficientes vs Alpha')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# Gráfico 2: Selección de variables
ax2.bar(variables, coeficientes_lasso[2, :], alpha=0.7, color=['blue', 'red', 'green', 'gray', 'gray'])
ax2.set_xlabel('Variables')
ax2.set_ylabel('Coeficiente (alpha=0.5)')
ax2.set_title('Selección de Variables por Lasso\n(Gris: variables eliminadas)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nCoeficientes con alpha=0.5:")
for var, coef in zip(variables, coeficientes_lasso[2, :]):
    print(f"{var}: {coef:.4f}")

print("\n→ Lasso elimina variables irrelevantes (Peajes, Mantenimiento)")
