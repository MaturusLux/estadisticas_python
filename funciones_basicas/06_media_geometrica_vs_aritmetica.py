# 06 Medía Aritmetica para determinar el punto de inflexión de incremento de gastos en Maquinaria X, a modo comparativo se determina la Aritmetica en el mismo ejercicio.
import matplotlib.pyplot as plt
import numpy as np  # Ya lo tienes

# Ej Orientado a Media Aritmetica
# Dato 1: Gasto Fijo Anual de Maquinaría ya en uso, gasto en Dolares.
data1 = [100, 200, 300, 400, 500, 600, 700, 800]

# Ej Orientado a Media Geometrica,
# Dato 2: Gastos Fijo en Maquinaría nueva, Media Geometrica para comparar y determinar punto de inflexión.
data2 = [100, 130, 190, 250, 340, 460, 630, 810]

# Calcular medias con numpy
data1 = np.array(data1)
arithmetic_mean = np.mean(data1)
#geometric_mean = np.exp(np.mean(np.log(data1)))  # Media geométrica

data2 = np.array(data2)
#arithmetic_mean = np.mean(data2)
geometric_mean = np.exp(np.mean(np.log(data2)))  # Media geométrica

# Gráfico
plt.figure(figsize=(8, 6))
plt.plot(data1, color='grey', label='Reparac. Maquin Usada')    # Línea azul para data1
plt.plot(data2, color='red', label='Reparac. Maquin Nueva')     # Línea roja para data2
plt.axhline(y=geometric_mean, color='r', linestyle='--', label=f'Media Geométrica: {geometric_mean:.2f}')
plt.axhline(y=arithmetic_mean, color='grey', linestyle=':', label=f'Media Aritmética: {arithmetic_mean:.2f}')
plt.title('Comparativa p/Considerar cambio de unidad (unica Variable: Gt. reparación)')
plt.xlabel('Index (Año)')
plt.ylabel('Value (Dolares x Miles)')
plt.legend()
plt.grid(True)
plt.show()

