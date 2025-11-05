import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# Regresión Ridge Lineal - Precio de flete
print("\n=== REGRESIÓN RIDGE LINEAL - PRECIO DE FLETE ===")

np.random.seed(42)
n_muestras = 100
# Nuevos datos para Ridge
km_campo = np.random.uniform(20, 300, n_muestras)
diesel_precio = np.random.uniform(800, 1200, n_muestras)
camion_antiguedad = np.random.uniform(1, 10, n_muestras)
costo_peajes = np.random.uniform(100, 400, n_muestras)
tiempo_viaje = np.random.uniform(2, 8, n_muestras)

# Precio del flete (relación multicolineal)
# Variables: km a campo, precio Gasoil, Costo Peaje, tiempo de viaje.
precio_flete = (
    0.5 * km_campo +
    0.3 * diesel_precio + 
    0.4 * costo_peajes +
    0.2 * tiempo_viaje +
    np.random.normal(0, 30, n_muestras)
)

# Matriz de diseño con multicolinealidad
X_ridge = np.column_stack([km_campo, diesel_precio, camion_antiguedad, costo_peajes, tiempo_viaje])
# Añadir variable correlacionada
X_ridge = np.column_stack([X_ridge, km_campo * 1.1 + np.random.normal(0, 10, n_muestras)])

X_ridge_std = (X_ridge - X_ridge.mean(axis=0)) / X_ridge.std(axis=0)
y_ridge_std = (precio_flete - precio_flete.mean()) / precio_flete.std()

# Función de costo Ridge
def ridge_cost(theta, X, y, alpha):
    m = len(y)
    predictions = X @ theta
    mse = np.sum((predictions - y) ** 2) / (2 * m)
    regularization = alpha * np.sum(theta ** 2) / 2
    return mse + regularization

# Gradiente de Ridge
def ridge_gradient(theta, X, y, alpha):
    m = len(y)
    predictions = X @ theta
    gradient = (X.T @ (predictions - y)) / m
    regularization = alpha * theta
    return gradient + regularization

# Optimización Ridge
alphas_ridge = [0, 0.1, 1, 10, 100]
coeficientes_ridge = []

for alpha in alphas_ridge:
    theta_inicial = np.zeros(X_ridge_std.shape[1])
    resultado = minimize(ridge_cost, theta_inicial, 
                        args=(X_ridge_std, y_ridge_std, alpha), 
                        method='BFGS',
                        jac=ridge_gradient)
    coeficientes_ridge.append(resultado.x)

coeficientes_ridge = np.array(coeficientes_ridge)

# Visualización Ridge
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico 1: Coeficientes Ridge vs Alpha
variables_ridge = ['KM', 'Diesel', 'Antiguedad', 'Peajes', 'Tiempo', 'KM_Correl']
for i in range(len(variables_ridge)):
    ax1.plot(alphas_ridge, coeficientes_ridge[:, i], marker='o', label=variables_ridge[i], linewidth=2)

ax1.set_xlabel('Alpha (fuerza de regularización)')
ax1.set_ylabel('Coeficientes')
ax1.set_title('Regularización Ridge: Coeficientes vs Alpha')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# Gráfico 2: Comparación coeficientes OLS vs Ridge
x_pos = np.arange(len(variables_ridge))
ancho = 0.35

ax2.bar(x_pos - ancho/2, coeficientes_ridge[0, :], ancho, label='OLS (alpha=0)', alpha=0.7)
ax2.bar(x_pos + ancho/2, coeficientes_ridge[2, :], ancho, label='Ridge (alpha=1)', alpha=0.7)

ax2.set_xlabel('Variables')
ax2.set_ylabel('Coeficientes')
ax2.set_title('Comparación: OLS vs Ridge\n(Reduce sobreajuste con multicolinealidad)')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(variables_ridge, rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nCoeficientes OLS vs Ridge (alpha=1):")
for var, ols, ridge in zip(variables_ridge, coeficientes_ridge[0, :], coeficientes_ridge[2, :]):
    print(f"{var}: OLS={ols:.4f}, Ridge={ridge:.4f}")

print("\n→ Ridge reduce coeficientes pero no los elimina")
