import numpy as np
import matplotlib.pyplot as plt

# Generar datos de ejemplo
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # 100 muestras, 1 feature
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + ruido

# Añadir término de bias (x0 = 1)
X_b = np.c_[np.ones((100, 1)), X]  # X_b = [1, X]

# Parámetros del algoritmo
learning_rate = 0.1
n_iterations = 1000
m = len(X_b)  # número de muestras

# Inicializar parámetros theta (pesos)
theta = np.random.randn(2, 1)

# Almacenar historial de costos
cost_history = []

# Gradiente Descendente por Lotes
for iteration in range(n_iterations):
    # Calcular gradientes (usando TODO el dataset)
    gradients = (2 / m) * X_b.T.dot(X_b.dot(theta) - y)

    # Actualizar parámetros
    theta = theta - learning_rate * gradients

    # Calcular y almacenar costo
    cost = (1 / m) * np.sum((X_b.dot(theta) - y) ** 2)
    cost_history.append(cost)

    # Mostrar progreso cada 100 iteraciones
    if iteration % 100 == 0:
        print(f"Iteración {iteration}: Costo = {cost:.4f}")

# Resultados finales
print(f"\nParámetros finales: theta0 = {theta[0][0]:.4f}, theta1 = {theta[1][0]:.4f}")
print(f"Ecuación: y = {theta[0][0]:.4f} + {theta[1][0]:.4f}*x")

# Visualización
plt.figure(figsize=(12, 4))

# Gráfico 1: Datos y línea de regresión
plt.subplot(1, 2, 1)
plt.scatter(X, y, alpha=0.7)
plt.plot(X, X_b.dot(theta), color='red', linewidth=2)
plt.title('Regresión Lineal con Gradiente Descendente')
plt.xlabel('X')
plt.ylabel('y')

# Gráfico 2: Evolución del costo
plt.subplot(1, 2, 2)
plt.plot(cost_history)
plt.title('Evolución del Costo')
plt.xlabel('Iteración')
plt.ylabel('Costo (MSE)')
plt.yscale('log')

plt.tight_layout()
plt.show()

# Output:
# Iteración 0: Costo = 11.7320
# Iteración 100: Costo = 0.8074
# Iteración 200: Costo = 0.8066
# Iteración 300: Costo = 0.8066
# Iteración 400: Costo = 0.8066
# Iteración 500: Costo = 0.8066
# Iteración 600: Costo = 0.8066
# Iteración 700: Costo = 0.8066
# Iteración 800: Costo = 0.8066
# Iteración 900: Costo = 0.8066
#
# Parámetros finales: theta0 = 4.2151, theta1 = 2.7701
# Ecuación: y = 4.2151 + 2.7701*x
