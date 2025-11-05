import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 1. Datos del bioensayo
log_dosis = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
n_insectos = np.array([100, 100, 100, 100, 100, 100, 100, 100])
muertos_observado = np.array([5, 12, 35, 60, 85, 96, 99, 100])

# 2. Preparar DataFrame
datos = pd.DataFrame({
    'log_dosis': log_dosis,
    'n_insectos': n_insectos,
    'muertos': muertos_observado
})

# 3. Matriz de predictores (con constante para el intercepto)
X = sm.add_constant(datos['log_dosis'])

# 4. Ajustar modelo GLM con distribución binomial y enlace probit
# Usamos la proporción como endog y los pesos como el número de ensayos
modelo_glm = sm.GLM(
    endog=datos['muertos'] / datos['n_insectos'],  # proporción de éxitos
    exog=X,
    family=sm.families.Binomial(link=sm.families.links.probit()),
    var_weights=datos['n_insectos']  # ¡esto es clave! Da peso a cada observación
)

# 5. Ajustar el modelo
resultados = modelo_glm.fit()

# 6. Mostrar resumen
print(resultados.summary())

# 7. Predicciones para curva suave
dosis_pred = np.linspace(log_dosis.min(), log_dosis.max(), 200)
X_pred = sm.add_constant(dosis_pred)
prediccion = resultados.predict(X_pred)

# 8. Gráfico
plt.figure(figsize=(10, 6))
plt.scatter(log_dosis, datos['muertos'] / datos['n_insectos'],
            color='darkgreen', s=60, label='Datos observados', zorder=5)
plt.plot(dosis_pred, prediccion, color='red', linewidth=2, label='Curva Probit (GLM)')
plt.xlabel('Logaritmo de la dosis de insecticida')
plt.ylabel('Proporción de insectos muertos')
plt.title('Regresión Probit con GLM (Bioensayo de Insecticida)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# 9. Calcular LD50 (dosis letal 50%)
# En probit: Φ⁻¹(0.5) = 0 → 0 = β₀ + β₁·log_dosis → log_dosis = -β₀/β₁
beta0 = resultados.params['const']
beta1 = resultados.params['log_dosis']
log_ld50 = -beta0 / beta1
ld50 = np.exp(log_ld50)

print(f"\n--- Resultado clave ---")
print(f"LD50 estimada: {ld50:.3f} unidades de dosis")

####################OUTPUT CONSOLA#####################
#                  Generalized Linear Model Regression Results
# ==============================================================================
# Dep. Variable:                      y   No. Observations:                    8
# Model:                            GLM   Df Residuals:                        6
# Model Family:                Binomial   Df Model:                            1
# Link Function:                 probit   Scale:                          1.0000
# Method:                          IRLS   Log-Likelihood:                -176.14
# Date:                Wed, 05 Nov 2025   Deviance:                       1.2105
# Time:                        01:09:24   Pearson chi2:                     1.16
# No. Iterations:                     7   Pseudo R-squ. (CS):              1.000
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const         -2.4905      0.167    -14.907      0.000      -2.818      -2.163
# log_dosis      1.3963      0.085     16.482      0.000       1.230       1.562
# ==============================================================================
#
# --- Resultado clave ---
# LD50 estimada: 5.952 unidades de dosis

