# Ejemplo: Selección de Ingeniero Agrónomo - 25 Preguntas Clave
# ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Configuración
np.random.seed(42)
plt.style.use('seaborn-v0_8')

# 1. SIMULACIÓN DE DATOS: 100 candidatos evaluados en 25 preguntas
n_candidatos = 100
n_preguntas = 25

# Crear nombres de preguntas específicas para Ingeniero Agrónomo
preguntas = [
    # Conocimientos Técnicos (5 preguntas)
    "Manejo_suelos_pampeanos", "Riego_eficiente", "Fertilizacion_NPK", 
    "Manejo_plagas_cultivos", "Biotecnologia_agricola",
    
    # Experiencia Práctica (5 preguntas)
    "Gestion_proyectos_agro", "Manejo_maquinaria_agricola", 
    "Logistica_cosecha_postcosecha", "Relacion_proveedores_clientes",
    "Gestion_riesgos_climaticos",
    
    # Habilidades Gerenciales (5 preguntas)
    "Liderazgo_equipos_campo", "Presupuestacion_control_costos",
    "Planificacion_estacional", "Negociacion_contratos",
    "Gestion_calidad_BPA",
    
    # Conocimiento Normativo (5 preguntas)
    "Legislacion_agropecuaria_ARG", "Normas_ambientales", 
    "Certificaciones_exportacion", "Seguridad_laboral_rural",
    "Regulaciones_fitosanitarias",
    
    # Adaptabilidad e Innovación (5 preguntas)
    "Adaptacion_cambio_climatico", "Tecnologia_agricola_precision",
    "Sustentabilidad_agroecologica", "Innovacion_procesos",
    "Resolucion_problemas_complejos"
]

# Simular matriz de evaluaciones (escala 1-10)
# Patrones de correlación simulados
def simular_evaluaciones_agronomos(n_candidatos, n_preguntas):
    datos = np.zeros((n_candidatos, n_preguntas))
    
    # Grupo 1: Conocimientos Técnicos (preguntas 0-4)
    factor_tecnico = np.random.normal(7, 1.5, n_candidatos)
    for i in range(5):
        datos[:, i] = factor_tecnico + np.random.normal(0, 0.5, n_candidatos)
    
    # Grupo 2: Experiencia Práctica (preguntas 5-9)
    factor_practico = np.random.normal(8, 1.2, n_candidatos)
    for i in range(5, 10):
        datos[:, i] = factor_practico + np.random.normal(0, 0.6, n_candidatos)
    
    # Grupo 3: Habilidades Gerenciales (preguntas 10-14)
    factor_gerencial = np.random.normal(6, 1.8, n_candidatos)
    for i in range(10, 15):
        datos[:, i] = factor_gerencial + np.random.normal(0, 0.7, n_candidatos)
    
    # Grupo 4: Conocimiento Normativo (preguntas 15-19)
    factor_normativo = np.random.normal(5, 2.0, n_candidatos)
    for i in range(15, 20):
        datos[:, i] = factor_normativo + np.random.normal(0, 0.8, n_candidatos)
    
    # Grupo 5: Adaptabilidad (preguntas 20-24)
    factor_adaptabilidad = np.random.normal(7, 1.3, n_candidatos)
    for i in range(20, 25):
        datos[:, i] = factor_adaptabilidad + np.random.normal(0, 0.5, n_candidatos)
    
    # Asegurar que esté en escala 1-10
    datos = np.clip(datos, 1, 10)
    return np.round(datos, 1)

# Generar datos simulados
datos_candidatos = simular_evaluaciones_agronomos(n_candidatos, n_preguntas)
df = pd.DataFrame(datos_candidatos, columns=preguntas)

print(" DATOS SIMULADOS - PRIMERAS 5 EVALUACIONES")
print(df.head())
print(f"\n Estadísticas descriptivas:")
print(df.describe().round(2))

# Análisis PCA para Identificar Preguntas Clave
# 2. ANÁLISIS PCA
def analisis_pca_preguntas(df, n_componentes=5):
    """Aplicar PCA para identificar preguntas más representativas"""
    
    # Estandarizar datos
    scaler = StandardScaler()
    datos_estandarizados = scaler.fit_transform(df)
    
    # Aplicar PCA
    pca = PCA(n_components=n_componentes)
    componentes_principales = pca.fit_transform(datos_estandarizados)
    
    # Crear DataFrame de componentes
    componentes_df = pd.DataFrame(
        pca.components_,
        columns=df.columns,
        index=[f'PC{i+1}' for i in range(n_componentes)]
    )
    
    return pca, componentes_df, componentes_principales

# Ejecutar PCA
pca, componentes_df, componentes_principales = analisis_pca_preguntas(df)

# Visualizar varianza explicada
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
varianza_acumulada = np.cumsum(pca.explained_variance_ratio_)
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
        pca.explained_variance_ratio_, alpha=0.7, label='Individual')
plt.plot(range(1, len(varianza_acumulada) + 1), varianza_acumulada, 
         'ro-', label='Acumulada')
plt.xlabel('Componente Principal')
plt.ylabel('Varianza Explicada')
plt.title('Varianza Explicada por Componentes PCA')
plt.legend()
plt.xticks(range(1, 6))

# Heatmap de contribuciones
plt.subplot(1, 3, 2)
sns.heatmap(componentes_df.T, cmap='RdBu_r', center=0, annot=True, 
            fmt='.2f', cbar_kws={'label': 'Contribución'})
plt.title('Contribución de Preguntas a Componentes PCA')
plt.ylabel('Preguntas')
plt.xlabel('Componentes Principales')

plt.tight_layout()
plt.show()

print(f" VARIANZA EXPLICADA:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {var:.3f} ({var*100:.1f}%)")
print(f"Total primeros 5 PC: {varianza_acumulada[4]:.3f} ({varianza_acumulada[4]*100:.1f}%)")

# Identificación de Preguntas Clave por PCA
# 3. IDENTIFICAR PREGUNTAS CLAVE SEGÚN PCA
def identificar_preguntas_clave_pca(componentes_df, n_preguntas_por_componente=3):
    """Identificar las preguntas más importantes según PCA"""
    
    preguntas_clave = {}
    
    for componente in componentes_df.index:
        # Tomar las preguntas con mayor contribución absoluta
        contribuciones = componentes_df.loc[componente].abs()
        top_preguntas = contribuciones.nlargest(n_preguntas_por_componente)
        
        preguntas_clave[componente] = {
            'preguntas': top_preguntas.index.tolist(),
            'contribuciones': top_preguntas.values.tolist(),
            'signo': [componentes_df.loc[componente, p] for p in top_preguntas.index]
        }
    
    return preguntas_clave

# Obtener preguntas clave
preguntas_clave_pca = identificar_preguntas_clave_pca(componentes_df)

print(" PREGUNTAS CLAVE IDENTIFICADAS POR PCA")
print("="*60)

for pc, info in preguntas_clave_pca.items():
    print(f"\n{pc}:")
    for i, (pregunta, contrib, signo) in enumerate(zip(info['preguntas'], 
                                                     info['contribuciones'], 
                                                     info['signo'])):
        direccion = "(+) Habilidad" if signo > 0 else "(-) Debilidad"
        print(f"  {i+1}. {pregunta}: {contrib:.3f} {direccion}")

# Comparación con ICA y NMF
# 4. COMPARACIÓN CON ICA Y NMF
def comparar_tecnicas_descomposicion(df, n_componentes=5):
    """Comparar PCA, ICA y NMF para identificar preguntas clave"""
    
    scaler = StandardScaler()
    datos_estandarizados = scaler.fit_transform(df)
    
    # Aplicar las tres técnicas
    pca = PCA(n_components=n_componentes, random_state=42)
    componentes_pca = pca.fit_transform(datos_estandarizados)
    
    ica = FastICA(n_components=n_componentes, random_state=42, max_iter=1000)
    componentes_ica = ica.fit_transform(datos_estandarizados)
    
    # NMF requiere datos no negativos
    datos_positivos = datos_estandarizados - datos_estandarizados.min() + 0.1
    nmf = NMF(n_components=n_componentes, random_state=42, max_iter=1000)
    componentes_nmf = nmf.fit_transform(datos_positivos)
    
    # Crear DataFrames de componentes
    df_pca = pd.DataFrame(pca.components_, columns=df.columns,
                         index=[f'PCA_{i+1}' for i in range(n_componentes)])
    
    df_ica = pd.DataFrame(ica.components_, columns=df.columns,
                         index=[f'ICA_{i+1}' for i in range(n_componentes)])
    
    df_nmf = pd.DataFrame(nmf.components_, columns=df.columns,
                         index=[f'NMF_{i+1}' for i in range(n_componentes)])
    
    return df_pca, df_ica, df_nmf

# Ejecutar comparación
df_pca, df_ica, df_nmf = comparar_tecnicas_descomposicion(df)

# Visualizar comparación
fig, axes = plt.subplots(2, 2, figsize=(20, 15))

# PCA
sns.heatmap(df_pca.T, ax=axes[0,0], cmap='RdBu_r', center=0, annot=False)
axes[0,0].set_title('PCA - Contribuciones de Preguntas')
axes[0,0].set_ylabel('Preguntas')

# ICA
sns.heatmap(df_ica.T, ax=axes[0,1], cmap='RdBu_r', center=0, annot=False)
axes[0,1].set_title('ICA - Componentes Independientes')
axes[0,1].set_ylabel('Preguntas')

# NMF
sns.heatmap(df_nmf.T, ax=axes[1,0], cmap='YlOrRd', annot=False)
axes[1,0].set_title('NMF - Factores No Negativos')
axes[1,0].set_ylabel('Preguntas')

# Top preguntas por técnica
top_pca = df_pca.abs().mean().nlargest(8)
top_ica = df_ica.abs().mean().nlargest(8)
top_nmf = df_nmf.mean().nlargest(8)

axes[1,1].barh(range(8), top_pca.values, alpha=0.7, label='PCA')
axes[1,1].barh(range(8), top_ica.values, alpha=0.7, label='ICA', left=top_pca.values)
axes[1,1].barh(range(8), top_nmf.values, alpha=0.7, label='NMF', 
               left=top_pca.values + top_ica.values)
axes[1,1].set_yticks(range(8))
axes[1,1].set_yticklabels(top_pca.index, fontsize=9)
axes[1,1].set_title('Top 8 Preguntas - Comparación Técnicas')
axes[1,1].legend()

plt.tight_layout()
plt.show()

# LISTA FINAL DE 25 PREGUNTAS ORGANIZADAS POR COMPONENTES
# 5. LISTA FINAL DE PREGUNTAS CLAVE ORGANIZADAS
def generar_lista_final_preguntas(df_pca, n_preguntas_total=25):
    """Generar lista final organizada de preguntas clave"""
    
    # Calcular importancia global de cada pregunta
    importancia_global = df_pca.abs().mean()
    preguntas_ordenadas = importancia_global.sort_values(ascending=False)
    
    print(" LISTA FINAL DE 25 PREGUNTAS CLAVE - INGENIERO AGRÓNOMO")
    print("="*70)
    print("\nOrganizadas por importancia según análisis PCA:")
    print("-" * 70)
    
    categorias = {
        "Técnicas": ["Manejo_suelos_pampeanos", "Riego_eficiente", "Fertilizacion_NPK", 
                    "Manejo_plagas_cultivos", "Biotecnologia_agricola"],
        "Prácticas": ["Gestion_proyectos_agro", "Manejo_maquinaria_agricola", 
                     "Logistica_cosecha_postcosecha", "Relacion_proveedores_clientes",
                     "Gestion_riesgos_climaticos"],
        "Gerenciales": ["Liderazgo_equipos_campo", "Presupuestacion_control_costos",
                       "Planificacion_estacional", "Negociacion_contratos",
                       "Gestion_calidad_BPA"],
        "Normativas": ["Legislacion_agropecuaria_ARG", "Normas_ambientales", 
                      "Certificaciones_exportacion", "Seguridad_laboral_rural",
                      "Regulaciones_fitosanitarias"],
        "Adaptabilidad": ["Adaptacion_cambio_climatico", "Tecnologia_agricola_precision",
                         "Sustentabilidad_agroecologica", "Innovacion_procesos",
                         "Resolucion_problemas_complejos"]
    }
    
    for i, (pregunta, importancia) in enumerate(preguntas_ordenadas.items(), 1):
        # Encontrar categoría
        categoria = "Otra"
        for cat, preguntas_cat in categorias.items():
            if pregunta in preguntas_cat:
                categoria = cat
                break
        
        print(f"{i:2d}. [{categoria:12}] {pregunta} (importancia: {importancia:.3f})")
    
    return preguntas_ordenadas

# Generar lista final
lista_final = generar_lista_final_preguntas(df_pca)

# 6. RECOMENDACIONES PARA EL PROCESO DE SELECCIÓN
print("\n" + "="*70)
print(" RECOMENDACIONES PRÁCTICAS PARA LA ENTREVISTA")
print("="*70)

print("""
1. **ENFOCARSE EN PC1**: Las preguntas del primer componente explican la mayor 
   variabilidad y discriminan mejor entre candidatos.

2. **BALANCE TEMÁTICO**: Incluir preguntas de cada categoría identificada:
   - Conocimientos técnicos específicos
   - Experiencia práctica en campo
   - Habilidades gerenciales
   - Conocimiento normativo argentino
   - Adaptabilidad e innovación

3. **PESOS DIFERENCIADOS**: Asignar mayor peso a preguntas con alta contribución
   en los primeros componentes principales.

4. **EVITAR REDUNDANCIAS**: Las preguntas muy correlacionadas miden lo mismo,
   seleccionar una representativa de cada grupo.

5. **VALIDACIÓN CONTINUA**: Recalcular periódicamente con nuevos datos para
   ajustar las preguntas según el mercado laboral actual.
""")

#######################################
# El análisis te permitirá identificar:

# Preguntas que mejor discriminan entre buenos y malos candidatos
# Grupos de preguntas redundantes que miden lo mismo
# Dimensiones subyacentes del perfil requerido
# Peso relativo de cada área de competencia
# Preguntas clave que maximizan la información obtenida
