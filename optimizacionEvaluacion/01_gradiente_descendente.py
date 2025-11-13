# Visualización del Gradiente Descendente, genérico, básico pero ilustrativo:
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualizar_gradiente_descendente():
    # Función: f(x,y) = x² + y²
    def funcion(x, y):
        return x**2 + y**2
    
    def gradiente(x, y):
        return np.array([2*x, 2*y])
    
    # Parámetros, tasa de aprendizaje, Iteraciones:
    learning_rate = 0.1
    n_iteraciones = 20
    punto_inicial = np.array([3.0, 4.0])
    
    # Gradiente descendente
    puntos = [punto_inicial]
    punto_actual = punto_inicial.copy()
    
    for i in range(n_iteraciones):
        grad = gradiente(punto_actual[0], punto_actual[1])
        punto_actual = punto_actual - learning_rate * grad
        puntos.append(punto_actual.copy())
    
    puntos = np.array(puntos)
    
    # Visualización
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = funcion(X, Y)
    
    fig = plt.figure(figsize=(15, 5))
    
    # Contorno 2D
    plt.subplot(1, 2, 1)
    plt.contour(X, Y, Z, levels=20)
    plt.plot(puntos[:, 0], puntos[:, 1], 'ro-', markersize=4)
    plt.title('Trayectoria del Gradiente Descendente - 2D')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    
    # Superficie 3D
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')
    ax.plot(puntos[:, 0], puntos[:, 1], funcion(puntos[:, 0], puntos[:, 1]), 
            'ro-', markersize=4, linewidth=2)
    ax.set_title('Trayectoria del Gradiente Descendente - 3D')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    
    plt.tight_layout()
    plt.show()
    
    return puntos

# Ejecutar visualización
trayectoria = visualizar_gradiente_descendente()
print("Trayectoria del gradiente:")
for i, punto in enumerate(trayectoria):
    print(f"Iteración {i}: ({punto[0]:.4f}, {punto[1]:.4f})")



    #
#   Parámetros Importantes
#   Tasa de Aprendizaje (Learning Rate) 

def comparar_learning_rates():
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    colores = ['red', 'blue', 'green', 'orange']
    
    plt.figure(figsize=(12, 8))
    
    for lr, color in zip(learning_rates, colores):
        # Ejecutar GD con diferente learning rate
        x, historial = gradiente_descendente(learning_rate=lr, n_iteraciones=50, x_inicial=5)
        
        plt.plot(historial, color=color, label=f'LR = {lr}', linewidth=2)
        plt.plot(len(historial)-1, historial[-1], 'o', color=color, markersize=8)
    
    plt.xlabel('Iteraciones')
    plt.ylabel('Valor de x')
    plt.title('Comparación de Diferentes Learning Rates')
    plt.legend()
    plt.grid(True)
    plt.show()

comparar_learning_rates()
# Ver opciones de implementar en el entrenamiento de modelos!
