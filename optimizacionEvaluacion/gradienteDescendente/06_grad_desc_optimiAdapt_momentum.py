import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Generar una función de costo con mínimo claro
def quadratic_cost(x, y):
    """Función cuadrática con mínimo en (2, 3)"""
    return (x - 2) ** 2 + (y - 3) ** 2 + 0.5 * np.sin(5 * x) * np.sin(5 * y)


def quadratic_cost_gradient(x, y):
    """Gradiente de la función de costo"""
    dx = 2 * (x - 2) + 2.5 * np.cos(5 * x) * np.sin(5 * y)
    dy = 2 * (y - 3) + 2.5 * np.sin(5 * x) * np.cos(5 * y)
    return np.array([dx, dy])


# Implementación de Gradient Descent con Momentum
class MomentumOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, params, grads):
        if self.velocity is None:
            self.velocity = np.zeros_like(params)

        # Actualizar velocidad (acumulando gradientes pasados)
        self.velocity = self.momentum * self.velocity - self.lr * grads

        # Actualizar parámetros
        params += self.velocity

        return params, self.velocity.copy()


# Implementación de GD sin Momentum para comparación
class SimpleGD:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, params, grads):
        params -= self.lr * grads
        return params, -self.lr * grads


# Función para visualizar la trayectoria CORREGIDA
def plot_optimization_trajectory(trajectory_momentum, trajectory_simple, cost_function):
    # Crear malla para el fondo
    x = np.linspace(-1, 5, 100)
    y = np.linspace(-1, 6, 100)
    X, Y = np.meshgrid(x, y)
    Z = cost_function(X, Y)

    fig = plt.figure(figsize=(20, 6))

    # Gráfico 2D
    ax1 = plt.subplot(1, 3, 1)
    contour = ax1.contour(X, Y, Z, levels=50, alpha=0.6)
    ax1.clabel(contour, inline=True, fontsize=8)

    # Trajectorias
    momentum_x = [p[0] for p in trajectory_momentum]
    momentum_y = [p[1] for p in trajectory_momentum]
    simple_x = [p[0] for p in trajectory_simple]
    simple_y = [p[1] for p in trajectory_simple]

    ax1.plot(momentum_x, momentum_y, 'o-', linewidth=2, markersize=4,
             label='Con Momentum', color='red', alpha=0.7)
    ax1.plot(simple_x, simple_y, 'o-', linewidth=2, markersize=4,
             label='Sin Momentum', color='blue', alpha=0.7)

    # Flechas de dirección CORREGIDAS
    for i in range(0, len(momentum_x) - 1, 3):
        ax1.arrow(momentum_x[i], momentum_y[i],
                  momentum_x[i + 1] - momentum_x[i], momentum_y[i + 1] - momentum_y[i],
                  head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.6)

    for i in range(0, len(simple_x) - 1, 3):
        ax1.arrow(simple_x[i], simple_y[i],
                  simple_x[i + 1] - simple_x[i], simple_y[i + 1] - simple_y[i],
                  head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.6)

    ax1.scatter([2], [3], color='green', s=200, marker='*', label='Mínimo (2,3)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Trayectoria de Optimización\nMomentum vs Sin Momentum')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gráfico 3D
    ax2 = plt.subplot(1, 3, 2, projection='3d')
    surf = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

    # Trajectorias en 3D
    momentum_z = [cost_function(p[0], p[1]) for p in trajectory_momentum]
    simple_z = [cost_function(p[0], p[1]) for p in trajectory_simple]

    ax2.plot(momentum_x, momentum_y, momentum_z, 'o-', linewidth=2,
             label='Con Momentum', color='red', alpha=0.8)
    ax2.plot(simple_x, simple_y, simple_z, 'o-', linewidth=2,
             label='Sin Momentum', color='blue', alpha=0.8)

    ax2.scatter([2], [3], [cost_function(2, 3)], color='green',
                s=200, marker='*', label='Mínimo')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Costo')
    ax2.set_title('Landscape 3D con Trayectorias')
    ax2.legend()

    # Gráfico de convergencia
    ax3 = plt.subplot(1, 3, 3)
    momentum_costs = [cost_function(p[0], p[1]) for p in trajectory_momentum]
    simple_costs = [cost_function(p[0], p[1]) for p in trajectory_simple]

    ax3.plot(momentum_costs, 'o-', linewidth=2, label='Con Momentum', color='red')
    ax3.plot(simple_costs, 'o-', linewidth=2, label='Sin Momentum', color='blue')
    ax3.set_xlabel('Iteración')
    ax3.set_ylabel('Costo')
    ax3.set_title('Convergencia del Costo')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return momentum_costs, simple_costs


# Simulación de optimización
def run_optimization_comparison():
    # Parámetros iniciales (los mismos para ambos)
    initial_params = np.array([-0.5, 5.5])

    # Optimizadores
    momentum_opt = MomentumOptimizer(learning_rate=0.05, momentum=0.9)
    simple_opt = SimpleGD(learning_rate=0.05)

    # Almacenar trayectorias
    trajectory_momentum = [initial_params.copy()]
    trajectory_simple = [initial_params.copy()]

    # Almacenar velocidades y actualizaciones
    velocities = []
    simple_updates = []

    print("=== OPTIMIZACIÓN CON MOMENTUM ===")
    print("Iniciando desde:", initial_params)
    print("\nIteración |  Con Momentum (X,Y)  |  Sin Momentum (X,Y)  |  Velocidad Momentum")
    print("-" * 85)

    params_momentum = initial_params.copy()
    params_simple = initial_params.copy()

    n_iterations = 30

    for i in range(n_iterations):
        # Calcular gradientes (iguales para ambos)
        grads = quadratic_cost_gradient(params_momentum[0], params_momentum[1])

        # Actualizar con momentum
        params_momentum, velocity = momentum_opt.update(params_momentum, grads)
        trajectory_momentum.append(params_momentum.copy())
        velocities.append(velocity)

        # Actualizar sin momentum
        params_simple, simple_update = simple_opt.update(params_simple, grads)
        trajectory_simple.append(params_simple.copy())
        simple_updates.append(simple_update)

        if i % 3 == 0:  # Mostrar cada 3 iteraciones
            print(f"{i:8d} | ({params_momentum[0]:6.3f}, {params_momentum[1]:6.3f}) | "
                  f"({params_simple[0]:6.3f}, {params_simple[1]:6.3f}) | "
                  f"({velocity[0]:6.3f}, {velocity[1]:6.3f})")

    return (np.array(trajectory_momentum), np.array(trajectory_simple),
            np.array(velocities), np.array(simple_updates))


# Ejecutar la simulación
print("DEMOSTRACIÓN DE CÓMO MOMENTUM DIRIGE CONSISTENTEMENTE HACIA EL MÍNIMO")
print("=" * 80)

trajectory_momentum, trajectory_simple, velocities, simple_updates = run_optimization_comparison()

# Visualizar resultados
momentum_costs, simple_costs = plot_optimization_trajectory(
    trajectory_momentum, trajectory_simple, quadratic_cost
)

# Análisis de las direcciones y consistencia
print("\n" + "=" * 80)
print("ANÁLISIS DE CONSISTENCIA EN DIRECCIONES")
print("=" * 80)


# Calcular cambios de dirección
def analyze_direction_consistency(trajectory):
    directions = []
    for i in range(1, len(trajectory) - 1):
        # Vector de dirección entre pasos consecutivos
        dir_vector = trajectory[i + 1] - trajectory[i]
        prev_dir_vector = trajectory[i] - trajectory[i - 1]

        # Coseno del ángulo entre direcciones consecutivas
        cos_angle = np.dot(dir_vector, prev_dir_vector) / (
                np.linalg.norm(dir_vector) * np.linalg.norm(prev_dir_vector) + 1e-8
        )
        directions.append(cos_angle)
    return np.array(directions)


momentum_directions = analyze_direction_consistency(trajectory_momentum)
simple_directions = analyze_direction_consistency(trajectory_simple)

print(f"\nConsistencia de dirección (1 = misma dirección, -1 = dirección opuesta):")
print(f"Momentum:  {np.mean(momentum_directions):.3f} ± {np.std(momentum_directions):.3f}")
print(f"Sin Momentum: {np.mean(simple_directions):.3f} ± {np.std(simple_directions):.3f}")

# Visualizar consistencia de direcciones
plt.figure(figsize=(15, 5))

# Gráfico 1: Consistencia de direcciones
plt.subplot(1, 3, 1)
plt.plot(momentum_directions, 'o-', label='Con Momentum', color='red', alpha=0.7)
plt.plot(simple_directions, 'o-', label='Sin Momentum', color='blue', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.xlabel('Iteración')
plt.ylabel('Consistencia de Dirección')
plt.title('Consistencia en Direcciones\n(Coseno del ángulo entre pasos)')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 2: Magnitud de actualizaciones
plt.subplot(1, 3, 2)
momentum_update_mags = [np.linalg.norm(v) for v in velocities]
simple_update_mags = [np.linalg.norm(u) for u in simple_updates]

plt.plot(momentum_update_mags, 'o-', label='Con Momentum', color='red')
plt.plot(simple_update_mags, 'o-', label='Sin Momentum', color='blue')
plt.xlabel('Iteración')
plt.ylabel('Magnitud de Actualización')
plt.title('Tamaño de los Pasos')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 3: Velocidad acumulada
plt.subplot(1, 3, 3)
momentum_velocity_x = [v[0] for v in velocities]
momentum_velocity_y = [v[1] for v in velocities]

plt.plot(momentum_velocity_x, label='Velocidad X', color='red', linestyle='--')
plt.plot(momentum_velocity_y, label='Velocidad Y', color='blue', linestyle='--')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.xlabel('Iteración')
plt.ylabel('Componentes de Velocidad')
plt.title('Componentes de Velocidad en Momentum')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Demostración de cómo momentum supera obstáculos
print("\n" + "=" * 80)
print("CÓMO MOMENTUM SUPERA OBSTÁCULOS Y DIRIGE CONSISTENTEMENTE")
print("=" * 80)

print("\n1. ✅ ACUMULACIÓN DE VELOCIDAD:")
print("   - Momentum: 'Recuerda' direcciones anteriores")
print("   - Sin Momentum: Cada paso es independiente")

print("\n2. ✅ DIRECCIÓN CONSISTENTE:")
print("   - Momentum: Mantiene dirección hacia el mínimo")
print("   - Sin Momentum: Cambia dirección frecuentemente")

print("\n3. ✅ SUPERACIÓN DE MÍNIMOS LOCALES:")
print("   - Momentum: La inercia ayuda a pasar por alto pequeños obstáculos")
print("   - Sin Momentum: Se puede quedar atrapado en mínimos locales")

print("\n4. ✅ CONVERGENCIA MÁS RÁPIDA:")
print(f"   - Momentum: Llega al mínimo en {len(momentum_costs)} iteraciones")
print(f"   - Sin Momentum: Convergencia más lenta o oscilante")

# Ejemplo adicional simplificado: Valle estrecho
print("\n" + "=" * 80)
print("EJEMPLO ADICIONAL: COMPORTAMIENTO EN DIFERENTES ESCENARIOS")
print("=" * 80)

# Probemos diferentes valores de momentum
momentum_values = [0.5, 0.9, 0.99]
colors = ['orange', 'red', 'purple']

plt.figure(figsize=(15, 4))

for j, momentum in enumerate(momentum_values):
    # Reinicializar para cada valor de momentum
    params = np.array([-0.5, 5.5])
    momentum_opt = MomentumOptimizer(learning_rate=0.05, momentum=momentum)
    trajectory = [params.copy()]

    for i in range(20):
        grads = quadratic_cost_gradient(params[0], params[1])
        params, _ = momentum_opt.update(params, grads)
        trajectory.append(params.copy())

    traj_x = [p[0] for p in trajectory]
    traj_y = [p[1] for p in trajectory]

    plt.subplot(1, 3, 1)
    plt.plot(traj_x, traj_y, 'o-', linewidth=2, markersize=3,
             label=f'Momentum={momentum}', color=colors[j], alpha=0.7)

    plt.subplot(1, 3, 2)
    costs = [quadratic_cost(p[0], p[1]) for p in trajectory]
    plt.plot(costs, 'o-', linewidth=2, label=f'Momentum={momentum}', color=colors[j])

    plt.subplot(1, 3, 3)
    # Calcular suavidad de la trayectoria
    if len(trajectory) > 2:
        smoothness = []
        for i in range(1, len(trajectory) - 1):
            vec1 = trajectory[i] - trajectory[i - 1]
            vec2 = trajectory[i + 1] - trajectory[i]
            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
            smoothness.append(cos_angle)
        plt.plot(smoothness, 'o-', linewidth=1, label=f'Momentum={momentum}', color=colors[j])

# Configurar los subplots
plt.subplot(1, 3, 1)
x = np.linspace(-1, 5, 100)
y = np.linspace(-1, 6, 100)
X, Y = np.meshgrid(x, y)
Z = quadratic_cost(X, Y)
contour = plt.contour(X, Y, Z, levels=20, alpha=0.3)
plt.scatter([2], [3], color='green', s=100, marker='*', label='Mínimo')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Trayectorias con Diferente Momentum')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.xlabel('Iteración')
plt.ylabel('Costo')
plt.title('Convergencia vs Momentum')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.xlabel('Paso')
plt.ylabel('Consistencia Dirección')
plt.title('Suavidad de Trayectoria')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("RESUMEN: CÓMO MOMENTUM DIRIGE CONSISTENTEMENTE")
print("=" * 80)

print("""
Momentum actúa como una 'bola inteligente' que:

1. 🎯 ACUMULA VELOCIDAD en direcciones consistentes
2. 📈 MANTIENE LA INERCIA cuando los gradientes son pequeños  
3. 🚀 ACELERA en regiones con pendiente constante
4. 🔄 SUAVIZA LAS OSCILACIONES en terrenos irregulares
5. 🎪 SUPERA OBSTÁCULOS gracias a la energía acumulada

¡Esto permite una convergencia más rápida y directa al mínimo!
""")
