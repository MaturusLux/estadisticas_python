# 02 Promedio Movil - Valor Moneda Extranjera.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Promedio Movil, datos Sintéticos.
np.random.seed(1)
t = np.arange(100)
data = 10 + 2 * np.sin(0.1 * t) + np.random.normal(0, 1, size=t.shape)
# Ventana de movilidad.
window = 10
moving_avg = pd.Series(data).rolling(window=window).mean()
# Gráfico Matplotlib.
plt.figure(figsize=(10, 4))
plt.plot(t, data, label='Datos Valor diario Mon. Extr.', alpha=0.6)
plt.plot(t, moving_avg, color='red', linewidth=2, label=f'Promedio móvil (ventana={window})')
plt.title('Promedio Móvil')
plt.xlabel('Tiempo en días')
plt.ylabel('Valor Mon. Extranjera.')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()
