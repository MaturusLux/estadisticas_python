# 01 Promedio - Rinde x Lote
import matplotlib.pyplot as plt
import numpy as np
#Promedio (Media aritmética)
ypoints = np.array([1.92, 2, 2.51, 1.7, 1.8, 1.5, 1.3])
xpoints = np.array(["Lote A", "Lote B", "Lote C", "Lote D", "Lote E", "Lote F", "Lote G"])
mean_val = np.mean(ypoints)
# Resultado x Consola.
print(f"\n=== PROMEDIO  ===")
print(f"Promedio: {mean_val:.3f}")
# Gráfico Matplolib.
plt.figure(figsize=(8, 4), facecolor='white')
plt.grid(True, axis='y', color='grey', linestyle='--', linewidth=0.5)
# Setting Background colour yellow
ax = plt.gca()
ax.set_facecolor("lightgrey")

plt.axhline(mean_val, color='red', linestyle='--', label=f'Promedio = {mean_val:.2f}')
plt.title('Promedio Rinde x Lote (Media Aritmética)')
plt.xlabel('LOTES')
plt.ylabel('Rinde x Lote')
plt.legend()

bars = plt.bar(xpoints, ypoints)
# Añadir etiquetas de valores
plt.bar_label(bars, fmt='%.2f T/ha')

plt.show()
