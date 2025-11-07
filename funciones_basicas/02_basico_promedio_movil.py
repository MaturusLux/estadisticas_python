# 02 - Promedio Movil - Cotizaciones
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Cargar datos desde CSV con dtype estructurado
dtype = [('date', 'datetime64[D]'), ('price', 'f8')]
data = np.genfromtxt('sojaCotizacion2025.csv', delimiter=',', names=True, dtype=dtype)

# Extraer fechas y precios
dates = data['date']
prices = data['price']

# Crear una Serie de pandas con las fechas como índice (opcional pero recomendado)
price_series = pd.Series(prices, index=dates)

# Calcular promedio móvil
window = 30
moving_avg = price_series.rolling(window=window).mean()

# Gráfico
plt.figure(figsize=(12, 6))
plt.plot(dates, prices, label='Precio Soja', alpha=0.5)
plt.plot(dates, moving_avg, label=f'Promedio Móvil ({window} días)', color='red', linewidth=2)
plt.title('Cotización de Soja con Promedio Móvil')
plt.xlabel('Fecha')
plt.ylabel('Precio (ARS)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()