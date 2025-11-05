import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import gammaln

# Datos REALES: Averías mensuales de tractores según horas de operación y antigüedad
# Variables:
# - horas_operacion: Horas mensuales de uso
# - antiguedad: Años de antigüedad del tractor
# - averias: Número de averías en el mes (count data)

datos = np.array([
    # horas, antiguedad, averias
    [80, 2, 1],
    [120, 1, 2],
    [150, 3, 3],
    [90, 1, 1],
    [200, 5, 6],
    [110, 2, 2],
    [180, 4, 5],
    [95, 1, 1],
    [160, 3, 4],
    [140, 2, 3],
    [170, 4, 4],
    [130, 2, 2],
    [190, 5, 5],
    [100, 1, 1],
    [210, 6, 7]
])

horas = datos[:, 0]
antiguedad = datos[:, 1]
averias = datos[:, 2]

print("=== DATOS DE AVERÍAS DE TRACTORES ===")
print(f"{'Horas':<8} {'Antiguedad':<12} {'Averías':<10}")
print("-" * 35)
for i in range(len(datos)):
    print(f"{horas[i]:<8} {antiguedad[i]:<12} {averias[i]:<10}")

# Preparar matriz de diseño (con intercepto)
X = np.column_stack([np.ones(len(horas)), horas, antiguedad])
y = averias

# FUNCIÓN DE LOG-VEROSIMILITUD PARA BINOMIAL NEGATIVA
def log_verosimilitud_binomial_negativa(params, X, y):
    """
    params: [beta0, beta1, beta2, alpha]
    alpha: parámetro de dispersión (alpha > 0)
    """
    beta = params[:-1]  # Coeficientes de regresión
    alpha = params[-1]  # Parámetro de dispersión
    
    # Predicción lineal
    mu = np.exp(X @ beta)
    
    # Parámetro r de la binomial negativa
    r = 1 / alpha
    
    # Log-verosimilitud
    log_lik = 0
    for i in range(len(y)):
        term1 = gammaln(y[i] + r) - gammaln(y[i] + 1) - gammaln(r)
        term2 = r * np.log(r) - r * np.log(r + mu[i])
        term3 = y[i] * np.log(mu[i]) - y[i] * np.log(r + mu[i])
        log_lik += term1 + term2 + term3
    
    return -log_lik  # Negativo para minimizar

# ESTIMACIÓN DE PARÁMETROS
print("\n=== ESTIMACIÓN DE PARÁMETROS ===")
print("Optimizando...")

# Valores iniciales
params_iniciales = np.array([0.0, 0.01, 0.1, 0.5])  # beta0, beta1, beta2, alpha

# Optimización
resultado = minimize(log_verosimilitud_binomial_negativa, params_iniciales, 
                    args=(X, y), method='BFGS')

if resultado.success:
    parametros_optimos = resultado.x
    beta_opt = parametros_optimos[:-1]
    alpha_opt = parametros_optimos[-1]
    
    print("¡Optimización exitosa!")
    print(f"Beta (intercepto): {beta_opt[0]:.4f}")
    print(f"Beta (horas): {beta_opt[1]:.4f}")
    print(f"Beta (antiguedad): {beta_opt[2]:.4f}")
    print(f"Alpha (dispersión): {alpha_opt:.4f}")
    print(f"Log-verosimilitud: {-resultado.fun:.4f}")
else:
    print("Error en la optimización")
    # Usar parámetros por defecto para continuar con el ejemplo
    beta_opt = np.array([-1.5, 0.02, 0.3])
    alpha_opt = 0.8

# PREDICCIONES
mu_pred = np.exp(X @ beta_opt)
r_pred = 1 / alpha_opt

print(f"\n=== PREDICCIONES ===")
print(f"{'Horas':<8} {'Antig.':<8} {'Averías Real':<15} {'Predicción':<12}")
print("-" * 50)
for i in range(len(datos)):
    print(f"{horas[i]:<8} {antiguedad[i]:<8} {averias[i]:<15} {mu_pred[i]:<12.2f}")

# VISUALIZACIÓN
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Gráfico 1: Averías vs Horas de operación
colores = antiguedad
scatter = ax1.scatter(horas, averias, c=colores, cmap='viridis', s=80, alpha=0.7)
ax1.set_xlabel('Horas de Operación Mensual')
ax1.set_ylabel('Número de Averías')
ax1.set_title('Averías vs Horas de Operación (Color: Antigüedad)')
plt.colorbar(scatter, ax=ax1, label='Antigüedad (años)')

# Línea de tendencia para antigüedad promedio
horas_range = np.linspace(80, 210, 100)
antiguedad_promedio = np.mean(antiguedad)
X_pred = np.column_stack([np.ones(100), horas_range, np.full(100, antiguedad_promedio)])
y_pred = np.exp(X_pred @ beta_opt)
ax1.plot(horas_range, y_pred, 'r-', linewidth=2, label='Tendencia (antig. promedio)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfico 2: Averías vs Antigüedad
scatter2 = ax2.scatter(antiguedad, averias, c=horas, cmap='plasma', s=80, alpha=0.7)
ax2.set_xlabel('Antigüedad (años)')
ax2.set_ylabel('Número de Averías')
ax2.set_title('Averías vs Antigüedad (Color: Horas de operación)')
plt.colorbar(scatter2, ax=ax2, label='Horas de operación')

# Línea de tendencia para horas promedio
antiguedad_range = np.linspace(1, 6, 100)
horas_promedio = np.mean(horas)
X_pred2 = np.column_stack([np.ones(100), np.full(100, horas_promedio), antiguedad_range])
y_pred2 = np.exp(X_pred2 @ beta_opt)
ax2.plot(antiguedad_range, y_pred2, 'r-', linewidth=2, label='Tendencia (horas promedio)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Gráfico 3: Valores observados vs predichos
ax3.scatter(averias, mu_pred, alpha=0.7, s=80)
ax3.plot([0, 8], [0, 8], 'r--', alpha=0.7, label='Línea perfecta')
ax3.set_xlabel('Averías Observadas')
ax3.set_ylabel('Averías Predichas')
ax3.set_title('Valores Observados vs Predichos')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Añadir etiquetas de puntos
for i, (obs, pred) in enumerate(zip(averias, mu_pred)):
    ax3.annotate(f'({horas[i]}h,{antiguedad[i]}a)', (obs, pred), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)

# Gráfico 4: Residuos
residuos = averias - mu_pred
ax4.scatter(mu_pred, residuos, alpha=0.7, s=80)
ax4.axhline(y=0, color='r', linestyle='--', alpha=0.7)
ax4.set_xlabel('Valores Predichos')
ax4.set_ylabel('Residuos')
ax4.set_title('Análisis de Residuos')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# INTERPRETACIÓN DE RESULTADOS
print(f"\n=== INTERPRETACIÓN DEL MODELO ===")
print("ECUACIÓN DEL MODELO:")
print(f"E[Averías] = exp({beta_opt[0]:.4f} + {beta_opt[1]:.4f}*horas + {beta_opt[2]:.4f}*antiguedad)")

print(f"\nEFECTOS MARGINALES:")
print(f"• Cada hora adicional aumenta las averías en un {np.exp(beta_opt[1])-1:.1%}")
print(f"• Cada año adicional aumenta las averías en un {np.exp(beta_opt[2])-1:.1%}")

print(f"\nPARÁMETRO DE DISPERSIÓN:")
print(f"Alpha = {alpha_opt:.4f}")
if alpha_opt > 0.5:
    print("→ Alta dispersión: Los datos tienen sobredispersión significativa")
    print("→ La Binomial Negativa es más apropiada que Poisson")
else:
    print("→ Dispersión moderada")

# PREDICCIÓN PARA NUEVOS CASOS
print(f"\n=== PREDICCIONES PARA NUEVOS TRACTORES ===")
nuevos_casos = np.array([
    [1, 100, 2],  # Tractor nuevo, uso moderado
    [1, 180, 5],  # Tractor viejo, uso intensivo
    [1, 140, 3]   # Tractor mediano, uso normal
])

predicciones_nuevos = np.exp(nuevos_casos @ beta_opt)

casos_desc = [
    "Nuevo (100h, 2años)",
    "Viejo (180h, 5años)", 
    "Mediano (140h, 3años)"
]

for desc, pred in zip(casos_desc, predicciones_nuevos):
    print(f"{desc}: {pred:.2f} averías esperadas")

# COMPARACIÓN CON POISSON (para mostrar ventajas de Binomial Negativa)
print(f"\n=== COMPARACIÓN CON MODELO POISSON ===")
print("En Poisson, la media es igual a la varianza")
varianza_observada = np.var(averias)
media_observada = np.mean(averias)
print(f"Media observada: {media_observada:.2f}")
print(f"Varianza observada: {varianza_observada:.2f}")
print(f"Razón varianza/media: {varianza_observada/media_observada:.2f}")

if varianza_observada > media_observada:
    print("→ SOBREDISPERSIÓN detectada (varianza > media)")
    print("→ La Binomial Negativa es más apropiada que Poisson")
else:
    print("→ No hay evidencia de sobredispersión")
