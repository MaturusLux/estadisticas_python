import numpy as np
import matplotlib.pyplot as plt

# Datos de averías de maquinarias
averias = np.array([
    'Tractor 1', 'Cosechadora 1', 'Tractor 1', 'Sembradora 1', 'Tractor 2',
    'Cosechadora 2', 'Tractor 1', 'Sembradora 2', 'Cosechadora 1', 'Tractor 2',
    'Tractor 1', 'Sembradora 1', 'Cosechadora 1', 'Tractor 2'
])

print("Lista completa de averías registradas:")
print(averias)
print(f"\nTotal de averías: {len(averias)}")

# Calcular la frecuencia de cada maquinaria
maquinarias, frecuencias = np.unique(averias, return_counts=True)

print("\nFrecuencia por maquinaria:")
for maq, freq in zip(maquinarias, frecuencias):
    print(f"{maq}: {freq} averías")

# Encontrar la moda (puede haber más de una)
max_frecuencia = np.max(frecuencias)
modas = maquinarias[frecuencias == max_frecuencia]

print(f"\n=== RESULTADO DE LA MODA ===")
print(f"Frecuencia máxima: {max_frecuencia} averías")
print(f"Moda(s): {', '.join(modas)}")

# Crear gráfico de barras
plt.figure(figsize=(12, 6))

# Gráfico de barras para todas las maquinarias
barras = plt.bar(maquinarias, frecuencias, color='skyblue', edgecolor='navy', alpha=0.7)

# Resaltar la(s) moda(s) en color diferente
for i, (maq, freq) in enumerate(zip(maquinarias, frecuencias)):
    if freq == max_frecuencia:
        barras[i].set_color('red')
        barras[i].set_alpha(0.8)

# Personalizar el gráfico
plt.title('Frecuencia de Averías por Maquinaria\n(Moda en rojo)', fontsize=14, fontweight='bold')
plt.xlabel('Tipo de Maquinaria', fontsize=12)
plt.ylabel('Número de Averías', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Añadir valores en las barras
for i, v in enumerate(frecuencias):
    plt.text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')

# Ajustar el layout para que no se corten las etiquetas
plt.tight_layout()

# Mostrar información adicional en el gráfico
plt.text(0.02, 0.98, f'Total de averías: {len(averias)}', 
         transform=plt.gca().transAxes, fontsize=10, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

plt.text(0.02, 0.90, f'Moda: {", ".join(modas)}', 
         transform=plt.gca().transAxes, fontsize=10, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))

plt.show()

# Análisis adicional
print("\n=== ANÁLISIS DETALLADO ===")
print("Distribución de averías:")
for maq, freq in zip(maquinarias, frecuencias):
    porcentaje = (freq / len(averias)) * 100
    print(f"{maq}: {freq} averías ({porcentaje:.1f}%)")

print(f"\nLa(s) maquinaria(s) con más averías es(son): {', '.join(modas)}")
print(f"Esto representa {max_frecuencia} de las {len(averias)} averías totales")
