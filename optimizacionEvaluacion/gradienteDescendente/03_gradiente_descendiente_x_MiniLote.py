import numpy as np
import matplotlib.pyplot as plt

# Generar datos de ejemplo
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)  # 1000 muestras
y = 4 + 3 * X + np.random.randn(1000, 1)  # y = 4 + 3x + ruido

# Añadir término de bias
X_b = np.c_[np.ones((1000, 1)), X]

# Parámetros del algoritmo
learning_rate = 0.1
n_epochs = 50
batch_size = 32
m = len(X_b)  # número total de muestras

# Inicializar parámetros
theta = np.random.randn(2, 1)

# Almacenar historial
cost_history = []
batch_count = m // batch_size


def compute_cost(X, y, theta):
    """Calcular costo MSE"""
    m = len(X)
    predictions = X.dot(theta)
    return (1 / m) * np.sum((predictions - y) ** 2)


# Gradiente Descendente por Mini-Lotes
for epoch in range(n_epochs):
    # Mezclar datos en cada época
    shuffled_indices = np.random.permutation(m)
    X_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]

    epoch_cost = 0

    # Procesar por mini-lotes
    for i in range(0, m, batch_size):
        # Obtener mini-lote
        end_index = min(i + batch_size, m)
        X_batch = X_shuffled[i:end_index]
        y_batch = y_shuffled[i:end_index]
        batch_m = len(X_batch)

        # Calcular gradiente del mini-lote
        gradients = (2 / batch_m) * X_batch.T.dot(X_batch.dot(theta) - y_batch)

        # Actualizar parámetros
        theta = theta - learning_rate * gradients

        # Calcular costo del mini-lote
        batch_cost = compute_cost(X_batch, y_batch, theta)
        epoch_cost += batch_cost

    # Costo promedio de la época
    avg_epoch_cost = epoch_cost / batch_count
    cost_history.append(avg_epoch_cost)

    if epoch % 10 == 0:
        print(f"Época {epoch}: Costo = {avg_epoch_cost:.4f}")

# Resultados finales
print(f"\nParámetros finales: theta0 = {theta[0][0]:.4f}, theta1 = {theta[1][0]:.4f}")
print(f"Ecuación: y = {theta[0][0]:.4f} + {theta[1][0]:.4f}*x")
print(f"Tamaño de mini-lote: {batch_size}")
print(f"Número de mini-lotes por época: {batch_count}")

# Visualización
plt.figure(figsize=(15, 4))

# Gráfico 1: Datos y línea de regresión
plt.subplot(1, 3, 1)
plt.scatter(X, y, alpha=0.3, s=20)
plt.plot(X, X_b.dot(theta), color='red', linewidth=2)
plt.title('Regresión con Mini-Lotes')
plt.xlabel('X')
plt.ylabel('y')

# Gráfico 2: Evolución del costo
plt.subplot(1, 3, 2)
plt.plot(cost_history)
plt.title('Evolución del Costo por Época')
plt.xlabel('Época')
plt.ylabel('Costo Promedio')
plt.yscale('log')

# Gráfico 3: Comparación de convergencia
plt.subplot(1, 3, 3)

# Comparar con Batch completo (referencia)
theta_batch = np.random.randn(2, 1)
batch_costs = []
for i in range(n_epochs):
    gradients = (2 / m) * X_b.T.dot(X_b.dot(theta_batch) - y)
    theta_batch = theta_batch - learning_rate * gradients
    cost = compute_cost(X_b, y, theta_batch)
    batch_costs.append(cost)

plt.plot(cost_history, label='Mini-Lotes')
plt.plot(batch_costs, label='Batch Completo', linestyle='--')
plt.title('Comparación: Mini-Lotes vs Batch')
plt.xlabel('Época/Iteración')
plt.ylabel('Costo')
plt.legend()
plt.yscale('log')

plt.tight_layout()
plt.show()

# Mostrar algunos mini-lotes de ejemplo
print("\nPrimeros 3 mini-lotes (primeras 5 muestras cada uno):")
for i in range(0, min(15, m), batch_size):
    end_index = min(i + 5, i + batch_size)  # Mostrar solo primeras 5
    print(f"Mini-lote {i//batch_size + 1}: muestras {i} a {end_index-1}")

# Época 0: Costo = 2.4565
# Época 10: Costo = 0.9793
# Época 20: Costo = 0.9845
# Época 30: Costo = 0.9915
# Época 40: Costo = 0.9871
# 
# Parámetros finales: theta0 = 4.0433, theta1 = 2.7692
# Ecuación: y = 4.0433 + 2.7692*x
# Tamaño de mini-lote: 32
# Número de mini-lotes por época: 31
#
# Primeros 3 mini-lotes (primeras 5 muestras cada uno):
# Mini-lote 1: muestras 0 a 4
