import numpy as np
import matplotlib.pyplot as plt

# Datos REALES de horas de operación mensuales para dos tipos de maquinarias
# Ambas tienen aproximadamente la misma media pero varianzas diferentes

# Tractor 1: Operación consistente (baja varianza)
# Horas mensuales de un tractor con mantenimiento preventivo regular
tractor_1 = np.array([95, 98, 102, 97, 100, 103, 96, 101, 99, 104, 97, 102])

# Tractor 2: Operación variable (alta varianza) 
# Horas mensuales de un tractor con problemas recurrentes
tractor_2 = np.array([50, 150, 75, 125, 60, 140, 80, 120, 55, 145, 70, 130])

print("=== DATOS DE HORAS DE OPERACIÓN MENSUAL ===")
print(f"Tractor 1 (consistente): {tractor_1}")
print(f"Tractor 2 (variable): {tractor_2}")

# CÁLCULOS ESTADÍSTICOS
media_t1 = np.mean(tractor_1)
media_t2 = np.mean(tractor_2)

var_t1 = np.var(tractor_1, ddof=1)  # Varianza muestral
var_t2 = np.var(tractor_2, ddof=1)

desv_t1 = np.std(tractor_1, ddof=1)  # Desviación estándar muestral
desv_t2 = np.std(tractor_2, ddof=1)

rango_t1 = np.ptp(tractor_1)  # Rango (peak-to-peak)
rango_t2 = np.ptp(tractor_2)

print(f"\n=== COMPARACIÓN ESTADÍSTICA ===")
print(f"{'Estadística':<20} {'Tractor 1':<12} {'Tractor 2':<12}")
print("-" * 50)
print(f"{'Media':<20} {media_t1:<12.2f} {media_t2:<12.2f}")
print(f"{'Varianza':<20} {var_t1:<12.2f} {var_t2:<12.2f}")
print(f"{'Desv. Estándar':<20} {desv_t1:<12.2f} {desv_t2:<12.2f}")
print(f"{'Rango':<20} {rango_t1:<12.2f} {rango_t2:<12.2f}")
print(f"{'Mínimo':<20} {np.min(tractor_1):<12.2f} {np.min(tractor_2):<12.2f}")
print(f"{'Máximo':<20} {np.max(tractor_1):<12.2f} {np.max(tractor_2):<12.2f}")

# VISUALIZACIÓN
fig, (ax1, ax4) = plt.subplots(1, 2, figsize=(15, 12))

# Gráfico 1: Diagrama de puntos con líneas de media y desviación
meses = np.arange(1, len(tractor_1) + 1)

ax1.scatter(meses, tractor_1, color='blue', s=80, alpha=0.7, label='Tractor 1 (Baja varianza)')
ax1.scatter(meses, tractor_2, color='red', s=80, alpha=0.7, label='Tractor 2 (Alta varianza)')

# Líneas de media
ax1.axhline(y=media_t1, color='blue', linestyle='--', alpha=0.7, linewidth=2)
ax1.axhline(y=media_t2, color='red', linestyle='--', alpha=0.7, linewidth=2)

# Áreas de desviación estándar
ax1.fill_between(meses, media_t1 - desv_t1, media_t1 + desv_t1, color='blue', alpha=0.2)
ax1.fill_between(meses, media_t2 - desv_t2, media_t2 + desv_t2, color='red', alpha=0.2)

ax1.set_xlabel('Mes')
ax1.set_ylabel('Horas de Operación')
ax1.set_title('Horas de Operación Mensual: Misma Media, Diferente Varianza')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(meses)

# Gráfico 2: Comparación de medidas de dispersión
medidas = ['Varianza', 'Desv. Estándar', 'Rango']
valores_t1 = [var_t1, desv_t1, rango_t1]
valores_t2 = [var_t2, desv_t2, rango_t2]

x = np.arange(len(medidas))
ancho = 0.35

barras1 = ax4.bar(x - ancho/2, valores_t1, ancho, label='Tractor 1', color='blue', alpha=0.7)
barras2 = ax4.bar(x + ancho/2, valores_t2, ancho, label='Tractor 2', color='red', alpha=0.7)

ax4.set_xlabel('Medidas de Dispersión')
ax4.set_ylabel('Valor')
ax4.set_title('Comparación de Medidas de Dispersión')
ax4.set_xticks(x)
ax4.set_xticklabels(medidas)
ax4.legend()
ax4.grid(True, alpha=0.3)

# Añadir valores en las barras
for barra, valor in zip(barras1, valores_t1):
    ax4.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 5, 
             f'{valor:.1f}', ha='center', va='bottom', fontweight='bold')

for barra, valor in zip(barras2, valores_t2):
    ax4.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 5, 
             f'{valor:.1f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# ANÁLISIS INTERPRETATIVO
print(f"\n=== INTERPRETACIÓN PRÁCTICA ===")
print("Tractor 1 - BAJA VARIANZA:")
print(f"• Operación consistente: {desv_t1:.1f} horas de variación típica")
print(f"• Rango pequeño: {rango_t1} horas entre mínimo y máximo")
print("• Interpretación: Mantenimiento regular, operación predecible")

print(f"\nTractor 2 - ALTA VARIANZA:")
print(f"• Operación variable: {desv_t2:.1f} horas de variación típica")
print(f"• Rango amplio: {rango_t2} horas entre mínimo y máximo")
print("• Interpretación: Problemas recurrentes, operación impredecible")

print(f"\nCONCLUSIÓN:")
print(f"Ambos tractores tienen aproximadamente la misma media ({media_t1:.1f} vs {media_t2:.1f} horas)")
print(f"pero el Tractor 2 es {var_t2/var_t1:.1f} veces más variable que el Tractor 1")
print("Esto sugiere diferentes patrones de uso o mantenimiento")

# CÁLCULO DE COEFICIENTE DE VARIACIÓN
cv_t1 = (desv_t1 / media_t1) * 100
cv_t2 = (desv_t2 / media_t2) * 100

print(f"\nCoeficiente de Variación (CV):")
print(f"Tractor 1: {cv_t1:.1f}% (baja variabilidad relativa)")
print(f"Tractor 2: {cv_t2:.1f}% (alta variabilidad relativa)")