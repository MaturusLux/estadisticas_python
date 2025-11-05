import sys
import matplotlib.pyplot as plt
import numpy as np  # Ya lo tienes

# Ej Orientado a Media Aritmetica
# Ejercicio 1: crecimiento lineal 10 Unidades, incorporación Cabezas de Vacunos por Año
#data = [10, 20, 30, 40, 50, 60, 70]

# Ej Orientado a Media Geometrica, 
# Ekjercicio 2: crecimiento natural del 34/36% de los animales sin vender, sin incorporar
# Datos: crecimiento natural del ganado (geométrico), y calcular la Media Geometrica
data = [10, 13, 19, 25, 34, 46, 63]

# Calcular medias con numpy
data = np.array(data)
arithmetic_mean = np.mean(data)
geometric_mean = np.exp(np.mean(np.log(data)))  # Media geométrica

# Gráfico
plt.figure(figsize=(8, 6))
plt.plot(data, 'o', label='Data Points')
plt.axhline(y=geometric_mean, color='r', linestyle='--', label=f'Geometric Mean: {geometric_mean:.2f}')
plt.axhline(y=arithmetic_mean, color='g', linestyle=':', label=f'Arithmetic Mean: {arithmetic_mean:.2f}')
plt.title('Data Points and Geometric Mean - Nacimiento Cabezas de Ganado (34/36% Anual)')
plt.xlabel('Index (Año)')
plt.ylabel('Value (Cabezas de Ganado)')
plt.legend()
plt.grid(True)
plt.show()
