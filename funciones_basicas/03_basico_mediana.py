# 03 - Mediana - Centro de Resultados
import numpy as np
import matplotlib.pyplot as plt

# Fijar semilla para reproducibilidad
np.random.seed(42)

# Simular lluvia en 20 lotes (en mm/año)
# 17 lotes con lluvia típica: entre 600 y 1000 mm
lluvia_normal = np.random.uniform(200, 600, size=17)

# 3 lotes con mucha más lluvia (ej. zonas montañosas): entre 1800 y 2500 mm
lluvia_alta = np.random.uniform(650, 1500, size=3)

# Combinar y mezclar
lluvia_total = np.concatenate([lluvia_normal, lluvia_alta])
# Mezclar para que los lotes con alta lluvia no estén al final
np.random.shuffle(lluvia_total)

# Calcular mediana
mediana_lluvia = np.median(lluvia_total)

# Crear gráfico de líneas (uno por lote)
lotes = np.arange(1, 21)  # Lotes 1 a 20

plt.figure(figsize=(12, 6))
plt.plot(lotes, lluvia_total, marker='o', linestyle='-', color='skyblue', linewidth=2, markersize=6, label='Lluvia por lote')
plt.axhline(mediana_lluvia, color='red', linestyle='--', linewidth=2, label=f'Mediana = {mediana_lluvia:.1f} mm')

# Resaltar los 3 lotes con más lluvia (opcional)
indices_altos = np.argsort(lluvia_total)[-3:]  # Índices de los 3 mayores
plt.scatter(lotes[indices_altos], lluvia_total[indices_altos], color='orange', s=100, zorder=5, label='Lluvia alta (3 lotes)')

# Personalización
plt.title('Milímetros de lluvia anual en 20 lotes\n(con 3 lotes de lluvia excepcional)', fontsize=14)
plt.xlabel('Lote')
plt.ylabel('Lluvia anual (mm)')
plt.xticks(lotes)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()

# Mostrar valor de la mediana
print(f"Mediana de lluvia anual en los 20 lotes: {mediana_lluvia:.1f} mm")
