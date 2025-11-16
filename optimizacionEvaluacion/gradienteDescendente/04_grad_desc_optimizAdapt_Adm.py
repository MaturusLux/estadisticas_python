import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

# Generar dataset más complejo
np.random.seed(42)
X, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)
y = y.reshape(-1, 1)

# Normalizar características
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Añadir término de bias
X_b = np.c_[np.ones((len(X_scaled), 1)), X_scaled]


# Implementación de Adam
class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # Primer momento (media)
        self.v = None  # Segundo momento (varianza)
        self.t = 0  # Paso de tiempo

    def update(self, params, grads):
        """Actualizar parámetros usando Adam"""
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        # Actualizar momentos
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        # Corrección de bias
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Actualizar parámetros
        params -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params


# Parámetros del entrenamiento
n_epochs = 200
batch_size = 32
m = len(X_b)

# Inicializar parámetros
theta = np.random.randn(6, 1) * 0.01  # 5 features + bias

# Crear optimizador Adam
adam = AdamOptimizer(learning_rate=0.01)

# Almacenar historial
cost_history = []
learning_rates = []  # Para trackear learning rate efectivo


def compute_cost(X, y, theta):
    """Calcular costo MSE"""
    m = len(X)
    predictions = X.dot(theta)
    return (1 / m) * np.sum((predictions - y) ** 2)


# Entrenamiento con Adam
for epoch in range(n_epochs):
    # Mezclar datos
    indices = np.random.permutation(m)
    X_shuffled = X_b[indices]
    y_shuffled = y_scaled[indices]

    epoch_cost = 0
    batch_count = 0

    for i in range(0, m, batch_size):
        # Obtener mini-lote
        end_index = min(i + batch_size, m)
        X_batch = X_shuffled[i:end_index]
        y_batch = y_shuffled[i:end_index]
        batch_m = len(X_batch)

        # Calcular gradiente
        predictions = X_batch.dot(theta)
        gradients = (2 / batch_m) * X_batch.T.dot(predictions - y_batch)

        # Actualizar parámetros con Adam
        theta = adam.update(theta, gradients)

        # Calcular costo
        batch_cost = compute_cost(X_batch, y_batch, theta)
        epoch_cost += batch_cost
        batch_count += 1

    # Guardar historial
    avg_cost = epoch_cost / batch_count
    cost_history.append(avg_cost)

    # Calcular learning rate efectivo (aproximado)
    if adam.t > 1:
        effective_lr = adam.lr * np.sqrt(1 - adam.beta2 ** adam.t) / (1 - adam.beta1 ** adam.t)
        learning_rates.append(effective_lr)

    if epoch % 20 == 0:
        print(f"Época {epoch:3d}: Costo = {avg_cost:.6f}")

# Resultados finales
print(f"\n--- Resultados con Adam ---")
print(f"Parámetros finales: {theta.flatten()}")
print(f"Costo final: {cost_history[-1]:.6f}")
print(f"Número total de actualizaciones: {adam.t}")


# Comparación con otros optimizadores
def train_with_optimizer(optimizer_type, learning_rate=0.01):
    """Función para comparar diferentes optimizadores"""
    theta_comp = np.random.randn(6, 1) * 0.01
    cost_history_comp = []

    if optimizer_type == 'sgd':
        # SGD simple
        for epoch in range(n_epochs):
            indices = np.random.permutation(m)
            X_shuffled = X_b[indices]
            y_shuffled = y_scaled[indices]

            epoch_cost = 0
            batch_count = 0

            for i in range(0, m, batch_size):
                end_index = min(i + batch_size, m)
                X_batch = X_shuffled[i:end_index]
                y_batch = y_shuffled[i:end_index]
                batch_m = len(X_batch)

                gradients = (2 / batch_m) * X_batch.T.dot(X_batch.dot(theta_comp) - y_batch)
                theta_comp -= learning_rate * gradients

                batch_cost = compute_cost(X_batch, y_batch, theta_comp)
                epoch_cost += batch_cost
                batch_count += 1

            cost_history_comp.append(epoch_cost / batch_count)

    return cost_history_comp


# Entrenar con SGD para comparación
sgd_costs = train_with_optimizer('sgd', learning_rate=0.01)

# Visualización
plt.figure(figsize=(15, 5))

# Gráfico 1: Evolución del costo
plt.subplot(1, 3, 1)
plt.plot(cost_history, label='Adam', linewidth=2)
plt.plot(sgd_costs, label='SGD', linestyle='--', alpha=0.7)
plt.title('Comparación: Adam vs SGD')
plt.xlabel('Época')
plt.ylabel('Costo')
plt.legend()
plt.yscale('log')

# Gráfico 2: Learning rate efectivo de Adam
plt.subplot(1, 3, 2)
plt.plot(learning_rates)
plt.title('Learning Rate Efectivo de Adam')
plt.xlabel('Actualización')
plt.ylabel('Learning Rate')

# Gráfico 3: Predicciones vs Valores reales
plt.subplot(1, 3, 3)
predictions = X_b.dot(theta)
plt.scatter(y_scaled, predictions, alpha=0.5)
plt.plot([y_scaled.min(), y_scaled.max()], [y_scaled.min(), y_scaled.max()], 'r--', linewidth=2)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs Reales')

plt.tight_layout()
plt.show()

# Mostrar métricas adicionales
final_predictions = X_b.dot(theta)
mse = np.mean((final_predictions - y_scaled) ** 2)
print(f"\n--- Métricas Finales ---")
print(f"MSE: {mse:.6f}")
print(f"R²: {1 - mse/np.var(y_scaled):.4f}")

# Probar diferentes learning rates
print(f"\n--- Sensibilidad a Learning Rate ---")
learning_rates_test = [0.001, 0.01, 0.1]
for lr in learning_rates_test:
    theta_test = np.random.randn(6, 1) * 0.01
    adam_test = AdamOptimizer(learning_rate=lr)

    # Una época rápida para ver comportamiento
    for i in range(0, min(100, m), batch_size):
        end_index = min(i + batch_size, m)
        X_batch = X_b[i:end_index]
        y_batch = y_scaled[i:end_index]

        gradients = (2 / len(X_batch)) * X_batch.T.dot(X_batch.dot(theta_test) - y_batch)
        theta_test = adam_test.update(theta_test, gradients)

    initial_cost = compute_cost(X_b[:100], y_scaled[:100], np.random.randn(6, 1) * 0.01)
    final_cost = compute_cost(X_b[:100], y_scaled[:100], theta_test)
    print(f"LR {lr}: Costo inicial {initial_cost:.4f} -> final {final_cost:.4f}")
