# 03 - Mediana - Centro de Recorridos de Reparto
import numpy as np
import matplotlib.pyplot as plt

# Mediana
# Datos manuales, distintas distancias de recorrido de Camión de Reparto de productos.
# Determinar el punto medio donde debería retornar en caso de no contar con cuenta corriente para carga gasoíl fuera de planta.
try:
    dtype = [('Destino', 'U20'), ('Kms', 'int')]
    data = np.genfromtxt('destinosReparto.csv', delimiter=',', names=True, dtype=dtype, encoding='utf-8')
    
    # Extraer destinos y kilómetros
    destinos = data['Destino']
    kms = data['Kms']
    
    # Calcular mediana de los kilómetros
    median_kms = np.median(kms)
    
    # Calcular estadísticas adicionales útiles
    total_kms = np.sum(kms)
    mean_kms = np.mean(kms)
    max_kms = np.max(kms)
    min_kms = np.min(kms)
    
    print(f"\n=== ANÁLISIS DE DISTANCIAS DE REPARTO ===")
    print(f"Mediana de distancias: {median_kms:.1f} km")
    print(f"Distancia promedio: {mean_kms:.1f} km")
    print(f"Distancia máxima: {max_kms} km")
    print(f"Distancia mínima: {min_kms} km")
    print(f"Suma total de kilómetros: {total_kms} km")
    print(f"Número de destinos: {len(destinos)}")
    
    # Mostrar destinos que están cerca de la mediana (para planificación de ruta)
    print(f"\nDestinos cercanos a la mediana ({median_kms:.1f} km):")
    for i, destino in enumerate(destinos):
        if abs(kms[i] - median_kms) <= 10:  # ±10 km de la mediana
            print(f"  - {destino}: {kms[i]} km")
    
    # Gráfico mejorado
    plt.figure(figsize=(10, 6))
    
    # Histograma
    plt.hist(kms, bins=15, alpha=0.7, color='lightgreen', edgecolor='black', label='Frecuencia de distancias')
    
    # Líneas de referencia
    plt.axvline(median_kms, color='darkgreen', linestyle='--', linewidth=2, 
                label=f'Mediana = {median_kms:.1f} km')
    plt.axvline(mean_kms, color='blue', linestyle=':', linewidth=2, 
                label=f'Promedio = {mean_kms:.1f} km')
    
    plt.title('Distribución de Distancias de Reparto', fontsize=14, fontweight='bold')
    plt.xlabel('Distancia del recorrido (km)', fontsize=12)
    plt.ylabel('Frecuencia de Destinos', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Añadir información en el gráfico
    plt.text(0.02, 0.98, f'Total destinos: {len(destinos)}\nSuma km: {total_kms}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
except FileNotFoundError:
    print("Error: No se encontró el archivo 'destinosReparto.csv'")
    print("Por favor, asegúrate de que el archivo existe en el directorio actual.")
    print("\nEstructura esperada del archivo CSV:")
    print("Destino,Kms")
    print("Cliente_A,45")
    print("Cliente_B,78")
    print("...")
    
except Exception as e:
    print(f"Error al procesar los datos: {e}")
