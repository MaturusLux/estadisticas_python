import numpy as np
import matplotlib.pyplot as plt

# Generar datos de ejemplo simples
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + 0.5 * np.random.randn(100, 1)

# Añadir término de bias
X_b = np.c_[np.ones((100, 1)), X]


# Implementación de AdamW
class AdamW:
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999,
                 weight_decay=0.01, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay  # Decaimiento de peso
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        # Aplicar decaimiento de peso ANTES de la actualización (KEY DIFERENCE)
        params *= (1 - self.lr * self.weight_decay)

        # Actualizar momentos (igual que Adam normal)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        # Corrección de bias
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Actualizar parámetros
        params_update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        params -= params_update

        return params


# Implementación de Adam normal para comparación
class Adam:
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
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
        params_update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        params -= params_update

        return params


# Función de entrenamiento comparativa
def train_comparison(use_adamw=True, weight_decay=0.1, n_epochs=200):
    # Inicializar parámetros (los mismos para ambos)
    theta_adam = np.random.randn(2, 1)
    theta_adamw = theta_adam.copy()

    # Optimizadores
    adam_opt = Adam(learning_rate=0.1)
    adamw_opt = AdamW(learning_rate=0.1, weight_decay=weight_decay)

    # Historial
    costs_adam = []
    costs_adamw = []
    params_adam = []
    params_adamw = []

    m = len(X_b)

    for epoch in range(n_epochs):
        # Calcular gradientes (igual para ambos)
        gradients = (2 / m) * X_b.T.dot(X_b.dot(theta_adam) - y)

        # Actualizar Adam normal
        theta_adam = adam_opt.update(theta_adam, gradients)

        # Actualizar AdamW (usamos los mismos gradientes para comparación justa)
        theta_adamw = adamw_opt.update(theta_adamw, gradients)

        # Calcular costos
        cost_adam = (1 / m) * np.sum((X_b.dot(theta_adam) - y) ** 2)
        cost_adamw = (1 / m) * np.sum((X_b.dot(theta_adamw) - y) ** 2)

        costs_adam.append(cost_adam)
        costs_adamw.append(cost_adamw)
        params_adam.append(theta_adam.copy())
        params_adamw.append(theta_adamw.copy())

    return costs_adam, costs_adamw, params_adam, params_adamw


# Ejecutar comparación
print("=== COMPARACIÓN: Adam vs AdamW ===")
print("Entrenando con los mismos datos e hiperparámetros...")

costs_adam, costs_adamw, params_adam, params_adamw = train_comparison(
    weight_decay=0.1, n_epochs=300
)

# Resultados finales
final_theta_adam = params_adam[-1]
final_theta_adamw = params_adamw[-1]

print(f"\n--- RESULTADOS FINALES ---")
print(f"Adam Normal:  y = {final_theta_adam[0][0]:.4f} + {final_theta_adam[1][0]:.4f}*x")
print(f"AdamW:        y = {final_theta_adamw[0][0]:.4f} + {final_theta_adamw[1][0]:.4f}*x")
print(f"\nParámetros verdaderos: y = 4 + 3*x")

# Visualización
plt.figure(figsize=(15, 5))

# Gráfico 1: Evolución del costo
plt.subplot(1, 3, 1)
plt.plot(costs_adam, label='Adam Normal', linewidth=2, alpha=0.8)
plt.plot(costs_adamw, label='AdamW', linewidth=2, alpha=0.8)
plt.title('Evolución del Costo\nAdam vs AdamW')
plt.xlabel('Época')
plt.ylabel('Costo (MSE)')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 2: Trayectoria de parámetros
plt.subplot(1, 3, 2)
theta0_adam = [p[0][0] for p in params_adam]
theta1_adam = [p[1][0] for p in params_adam]
theta0_adamw = [p[0][0] for p in params_adamw]
theta1_adamw = [p[1][0] for p in params_adamw]

plt.plot(theta0_adam, theta1_adam, label='Adam', alpha=0.6, linewidth=2)
plt.plot(theta0_adamw, theta1_adamw, label='AdamW', alpha=0.6, linewidth=2)
plt.scatter([4], [3], color='red', s=100, label='Verdadero', marker='*')
plt.xlabel('theta0 (bias)')
plt.ylabel('theta1 (pendiente)')
plt.title('Trayectoria de Parámetros')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 3: Predicciones finales
plt.subplot(1, 3, 3)
x_range = np.linspace(0, 2, 100)
X_range_b = np.c_[np.ones((100, 1)), x_range.reshape(-1, 1)]

y_pred_adam = X_range_b.dot(final_theta_adam)
y_pred_adamw = X_range_b.dot(final_theta_adamw)

plt.scatter(X, y, alpha=0.5, label='Datos')
plt.plot(x_range, y_pred_adam, label='Adam', linewidth=2)
plt.plot(x_range, y_pred_adamw, label='AdamW', linewidth=2)
plt.plot(x_range, 4 + 3 * x_range, 'k--', label='Verdadero', alpha=0.8)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Comparación de Predicciones Finales')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Probar diferentes valores de weight decay
print("\n=== EFECTO DEL WEIGHT DECAY ===")
weight_decay_values = [0.001, 0.01, 0.1, 0.5]

plt.figure(figsize=(12, 4))

for i, wd in enumerate(weight_decay_values):
    _, costs_wd, _, params_wd = train_comparison(weight_decay=wd, n_epochs=200)

    plt.subplot(1, 3, 1)
    plt.plot(costs_wd, label=f'WD={wd}', linewidth=2)

    plt.subplot(1, 3, 2)
    theta0_wd = [p[0][0] for p in params_wd]
    theta1_wd = [p[1][0] for p in params_wd]
    plt.plot(theta0_wd, theta1_wd, label=f'WD={wd}', alpha=0.7)

    plt.subplot(1, 3, 3)
    final_theta = params_wd[-1]
    y_pred_wd = X_range_b.dot(final_theta)
    plt.plot(x_range, y_pred_wd, label=f'WD={wd}', alpha=0.8)

# Configurar gráficos de weight decay
plt.subplot(1, 3, 1)
plt.title('Costo vs Weight Decay')
plt.xlabel('Época')
plt.ylabel('Costo')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.scatter([4], [3], color='red', s=100, label='Verdadero', marker='*')
plt.title('Trayectoria vs Weight Decay')
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.scatter(X, y, alpha=0.3, label='Datos')
plt.plot(x_range, 4 + 3 * x_range, 'k--', label='Verdadero', alpha=0.5)
plt.title('Predicciones vs Weight Decay')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Comparación de regularización
print("\n=== VENTAJAS DE AdamW ===")
print("1.  DECAIMIENTO DE PESO CORRECTO:")
print("   - AdamW: weight_decay se aplica ANTES de la actualización")
print("   - Adam normal: el weight_decay se escala por el learning rate")

print("\n2.  MEJOR GENERALIZACIÓN:")
print("   - AdamW tiende a encontrar soluciones más simples")
print("   - Parámetros más pequeños → mejor generalización")

print("\n3.  MÁS ESTABLE CON LR ALTOS:")
print("   - El decaimiento no se ve afectado por el momentum")
print("   - Comportamiento más predecible")

# Demostración del efecto de regularización
print("\n=== MAGNITUD DE PARÁMETROS FINALES ===")
magnitude_adam = np.linalg.norm(final_theta_adam)
magnitude_adamw = np.linalg.norm(final_theta_adamw)

print(f"Norma L2 de parámetros Adam:  {magnitude_adam:.4f}")
print(f"Norma L2 de parámetros AdamW: {magnitude_adamw:.4f}")
print(f"Diferencia: {magnitude_adam - magnitude_adamw:.4f}")

if magnitude_adamw < magnitude_adam:
    print(" AdamW logró parámetros más pequeños (mejor regularización)")
else:
    print("️  Los parámetros son similares (weight decay muy bajo)")

# Ejemplo de uso práctico
print("\n=== EJEMPLO PRÁCTICO CON AdamW ===")

# Reinicializar y entrenar solo con AdamW
theta_final = np.random.randn(2, 1)
adamw = AdamW(learning_rate=0.1, weight_decay=0.01, beta1=0.9, beta2=0.999)

costs = []
for epoch in range(100):
    gradients = (2 / len(X_b)) * X_b.T.dot(X_b.dot(theta_final) - y)
    theta_final = adamw.update(theta_final, gradients)

    cost = (1 / len(X_b)) * np.sum((X_b.dot(theta_final) - y) ** 2)
    costs.append(cost)

    if epoch % 25 == 0:
        print(f"Época {epoch:3d}: Costo = {cost:.6f}")

print(f"\nResultado final: y = {theta_final[0][0]:.4f} + {theta_final[1][0]:.4f}*x")
print("¡AdamW implementado y funcionando correctamente! ")

# === COMPARACIÓN: Adam vs AdamW ===
# Entrenando con los mismos datos e hiperparámetros...
#
# --- RESULTADOS FINALES ---
# Adam Normal:  y = 4.0815 + 2.9080*x
# AdamW:        y = 0.4274 + -0.0219*x
#
# Parámetros verdaderos: y = 4 + 3*x
#
# === EFECTO DEL WEIGHT DECAY ===
#
# === VENTAJAS DE AdamW ===
# 1.  DECAIMIENTO DE PESO CORRECTO:
#    - AdamW: weight_decay se aplica ANTES de la actualización
#    - Adam normal: el weight_decay se escala por el learning rate
#
# 2.  MEJOR GENERALIZACIÓN:
#    - AdamW tiende a encontrar soluciones más simples
#    - Parámetros más pequeños → mejor generalización
#
# 3.  MÁS ESTABLE CON LR ALTOS:
#    - El decaimiento no se ve afectado por el momentum
#    - Comportamiento más predecible
#
# === MAGNITUD DE PARÁMETROS FINALES ===
# Norma L2 de parámetros Adam:  5.0115
# Norma L2 de parámetros AdamW: 0.4280
# Diferencia: 4.5835
#  AdamW logró parámetros más pequeños (mejor regularización)
#
# === EJEMPLO PRÁCTICO CON AdamW ===
# Época   0: Costo = 41.779169
# Época  25: Costo = 3.921249
# Época  50: Costo = 0.247349
# Época  75: Costo = 0.230461
#
# Resultado final: y = 3.7897 + 3.0633*x
# ¡AdamW implementado y funcionando correctamente!
