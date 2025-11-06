import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Descargar datos de múltiples commodities (enero 2023 - febrero 2024)
print("=== DBSCAN CLUSTERING - DETECCIÓN DE PATRONES Y ANOMALÍAS ===")

tickers = {
    # Commodities Agrícolas
    'SOJA': 'ZS=F',
    'MAIZ': 'ZC=F', 
    'TRIGO': 'ZW=F',
    'AZUCAR': 'SB=F',
    'CAFE': 'KC=F',
    'ALGODON': 'CT=F',
    
    # Energéticos
    'PETROLEO': 'CL=F',
    'GAS_NATURAL': 'NG=F',
    
    # Metales
    'ORO': 'GC=F',
    'PLATA': 'SI=F',
    'COBRE': 'HG=F',
    
    # Ganadería
    'GANADO_VIVO': 'LE=F',
    'CERDO_MAGRO': 'HE=F'
}

# Descargar datos
print("Descargando datos de Yahoo Finance...")
datos = {}
for nombre, ticker in tickers.items():
    try:
        data = yf.download(ticker, start='2023-01-01', end='2024-03-01')
        datos[nombre] = data['Adj Close']
        print(f" {nombre}: {len(data)} días")
    except Exception as e:
        print(f" Error con {nombre}: {e}")

# Crear DataFrame
df = pd.DataFrame(datos)
df = df.dropna()

print(f"\nDatos descargados: {df.shape[0]} días trading")
print(f"Commodities: {list(df.columns)}")

# Calcular características para DBSCAN
print("\nCalculando características para clustering...")
caracteristicas = pd.DataFrame()

# 1. Retornos diarios (comportamiento inmediato)
for commodity in df.columns:
    caracteristicas[f'{commodity}_Retorno'] = df[commodity].pct_change()

# 2. Volatilidad rolling 5 días (riesgo a corto plazo)
for commodity in df.columns:
    caracteristicas[f'{commodity}_Volatilidad'] = df[commodity].pct_change().rolling(5).std()

# 3. Fuerza relativa rolling 10 días (tendencia)
for commodity in df.columns:
    caracteristicas[f'{commodity}_Fuerza_Relativa'] = df[commodity].pct_change(10)

# 4. Correlación móvil Soja-Petróleo (relación agrícola-energética)
caracteristicas['Correlacion_Soja_Petroleo'] = df[['SOJA', 'PETROLEO']].pct_change().rolling(10).corr().iloc[0::2, 1].values

# 5. Ratio Oro/Plata (sentimiento metales preciosos)
caracteristicas['Ratio_Oro_Plata'] = df['ORO'] / df['PLATA']

# 6. Spread Agrícola-Energético
caracteristicas['Spread_Agricola_Energia'] = (df[['SOJA', 'MAIZ', 'TRIGO']].mean(axis=1) / df[['PETROLEO', 'GAS_NATURAL']].mean(axis=1))

# Eliminar NaN y preparar datos
caracteristicas = caracteristicas.dropna()
X = caracteristicas.values

print(f"Características calculadas: {X.shape}")
print(f"Número de dimensiones: {X.shape[1]}")

# Estandarizar datos (CRÍTICO para DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# APLICAR DBSCAN
print("\nAplicando DBSCAN...")
dbscan = DBSCAN(
    eps=2.5,           # Distancia máxima entre puntos del mismo cluster
    min_samples=5,     # Mínimo puntos para formar cluster
    metric='euclidean' # Métrica de distancia
)

clusters = dbscan.fit_predict(X_scaled)

# Análisis de resultados
unique_clusters = np.unique(clusters)
n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
n_ruido = np.sum(clusters == -1)

print(f"\n RESULTADOS DBSCAN:")
print(f"• Clusters identificados: {n_clusters}")
print(f"• Puntos considerados ruido/anomalías: {n_ruido} ({n_ruido/len(clusters)*100:.1f}%)")
print(f"• Distribución de clusters: {np.unique(clusters, return_counts=True)}")

# Añadir clusters al DataFrame
df_clusters = df.iloc[len(df) - len(clusters):].copy()
df_clusters['Cluster'] = clusters
df_clusters['Es_Anomalia'] = clusters == -1

# VISUALIZACIONES DBSCAN
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

# Gráfico 1: Proyección 2D con PCA para visualización
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

colores = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))
for i, cluster_id in enumerate(unique_clusters):
    if cluster_id == -1:
        # Ruido/Anomalías en negro
        mask = clusters == cluster_id
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c='black', marker='x', s=50, label='Anomalías', alpha=0.8)
    else:
        mask = clusters == cluster_id
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=[colores[i]], s=40, label=f'Cluster {cluster_id}', alpha=0.7)

ax1.set_xlabel('Componente Principal 1')
ax1.set_ylabel('Componente Principal 2')
ax1.set_title('DBSCAN - Visualización PCA\n(Clusters basados en densidad)')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)

# Gráfico 2: Relación Soja vs Petróleo coloreado por clusters
for cluster_id in unique_clusters:
    if cluster_id == -1:
        mask = clusters == cluster_id
        ax2.scatter(df_clusters.loc[mask, 'SOJA'], df_clusters.loc[mask, 'PETROLEO'],
                   c='black', marker='x', s=50, label='Anomalías', alpha=0.8)
    else:
        mask = clusters == cluster_id
        ax2.scatter(df_clusters.loc[mask, 'SOJA'], df_clusters.loc[mask, 'PETROLEO'],
                   c=[colores[cluster_id]], s=40, label=f'Cluster {cluster_id}', alpha=0.7)

ax2.set_xlabel('Precio Soja (USD)')
ax2.set_ylabel('Precio Petróleo (USD)')
ax2.set_title('Relación Soja-Petróleo por Cluster DBSCAN')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Gráfico 3: Evolución temporal mostrando anomalías
fechas = df_clusters.index
for cluster_id in unique_clusters:
    if cluster_id == -1:
        mask = clusters == cluster_id
        cluster_fechas = fechas[mask]
        ax3.scatter(cluster_fechas, [0] * len(cluster_fechas), 
                   c='red', marker='x', s=100, label='Días Anómalos', alpha=0.8)
    else:
        mask = clusters == cluster_id
        cluster_fechas = fechas[mask]
        ax3.scatter(cluster_fechas, [cluster_id] * len(cluster_fechas), 
                   c=[colores[cluster_id]], s=40, label=f'Cluster {cluster_id}', alpha=0.7)

ax3.set_xlabel('Fecha')
ax3.set_ylabel('Cluster')
ax3.set_title('Evolución Temporal - DBSCAN Detecta Días Anómalos')
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# Gráfico 4: Comportamiento de clusters en metales preciosos
for cluster_id in unique_clusters:
    if cluster_id == -1:
        mask = clusters == cluster_id
        ax4.scatter(df_clusters.loc[mask, 'ORO'], df_clusters.loc[mask, 'PLATA'],
                   c='black', marker='x', s=60, label='Anomalías', alpha=0.8)
    else:
        mask = clusters == cluster_id
        ax4.scatter(df_clusters.loc[mask, 'ORO'], df_clusters.loc[mask, 'PLATA'],
                   c=[colores[cluster_id]], s=40, label=f'Cluster {cluster_id}', alpha=0.7)

ax4.set_xlabel('Precio Oro (USD)')
ax4.set_ylabel('Precio Plata (USD)')
ax4.set_title('Metales Preciosos - Patrones por Cluster')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ANÁLISIS DETALLADO DE CLUSTERS Y ANOMALÍAS
print("\n" + "="*70)
print("ANÁLISIS DETALLADO DE CLUSTERS DBSCAN")
print("="*70)

# Caracterizar cada cluster
for cluster_id in unique_clusters:
    if cluster_id == -1:
        cluster_data = df_clusters[df_clusters['Cluster'] == cluster_id]
        print(f"\n ANOMALÍAS DETECTADAS ({len(cluster_data)} días):")
        
        # Mostrar las 5 anomalías más recientes
        anomalias_recientes = cluster_data.tail(5)
        for fecha, fila in anomalias_recientes.iterrows():
            print(f"    {fecha.strftime('%Y-%m-%d')}:")
            print(f"      Soja: ${fila['SOJA']:.2f}, Petróleo: ${fila['PETROLEO']:.2f}")
            print(f"      Oro: ${fila['ORO']:.2f}, Maíz: ${fila['MAIZ']:.2f}")
    else:
        cluster_data = df_clusters[df_clusters['Cluster'] == cluster_id]
        
        print(f"\n CLUSTER {cluster_id} ({len(cluster_data)} días):")
        print(f"   Rango: {cluster_data.index.min().strftime('%Y-%m-%d')} a {cluster_data.index.max().strftime('%Y-%m-%d')}")
        
        # Comportamiento promedio del cluster
        retorno_promedio = cluster_data[['SOJA', 'MAIZ', 'PETROLEO', 'ORO']].pct_change().mean().mean()
        volatilidad_promedio = cluster_data[['SOJA', 'MAIZ', 'PETROLEO', 'ORO']].pct_change().std().mean()
        
        print(f"   Retorno promedio: {retorno_promedio:.4f}")
        print(f"   Volatilidad promedio: {volatilidad_promedio:.4f}")

# ANÁLISIS DE ANOMALÍAS ESPECÍFICAS
print("\n" + "="*70)
print("INVESTIGACIÓN DE ANOMALÍAS DETECTADAS")
print("="*70)

anomalias = df_clusters[df_clusters['Es_Anomalia']]
if len(anomalias) > 0:
    print(f"Se detectaron {len(anomalias)} días anómalos:")
    
    for fecha, fila in anomalias.tail(10).iterrows():  # Últimas 10 anomalías
        print(f"\n {fecha.strftime('%Y-%m-%d')} - Posibles causas:")
        
        # Analizar comportamientos extremos
        comportamientos = []
        
        # Verificar movimientos extremos
        retornos_dia = df.pct_change().loc[fecha].abs()
        commodities_extremos = retornos_dia.nlargest(3)
        
        for commodity, retorno in commodities_extremos.items():
            if retorno > 0.03:  # 3% de movimiento
                direccion = "SUBIO" if df.pct_change().loc[fecha][commodity] > 0 else "BAJÓ"
                comportamientos.append(f"{commodity} {direccion} {retorno:.1%}")
        
        if comportamientos:
            print(f"   Movimientos extremos: {', '.join(comportamientos)}")
        
        # Verificar correlaciones inusuales
        correlacion_soja_petroleo = df[['SOJA', 'PETROLEO']].pct_change().rolling(5).corr().loc[fecha].iloc[0,1]
        if abs(correlacion_soja_petroleo) < 0.1:
            print(f"   Correlación Soja-Petróleo inusual: {correlacion_soja_petroleo:.3f}")

# PATRONES INTER-MERCADOS IDENTIFICADOS
print("\n" + "="*70)
print("PATRONES INTER-MERCADOS DETECTADOS")
print("="*70)

# Analizar relaciones entre clusters de diferentes commodities
for cluster_id in unique_clusters:
    if cluster_id != -1:
        cluster_data = df_clusters[df_clusters['Cluster'] == cluster_id]
        
        print(f"\n CLUSTER {cluster_id}:")
        
        # Correlación agrícolas-energéticos
        correl_agricola_energia = cluster_data[['SOJA', 'MAIZ', 'PETROLEO', 'GAS_NATURAL']].corr()
        correl_promedio = (correl_agricola_energia.loc[['SOJA','MAIZ'], ['PETROLEO','GAS_NATURAL']].values.mean())
        
        # Correlación metales
        correl_metales = cluster_data[['ORO', 'PLATA', 'COBRE']].corr().mean().mean()
        
        print(f"  • Correlación Agrícola-Energía: {correl_promedio:.3f}")
        print(f"  • Correlación Metales: {correl_metales:.3f}")
        
        # Identificar patrón
        if correl_promedio > 0.6:
            print(f"  → PATRÓN: MERCADOS ALTAMENTE COORDINADOS")
        elif correl_promedio < 0.2:
            print(f"  → PATRÓN: MERCADOS DESACOPLADOS")
        
        if correl_metales > 0.7:
            print(f"  → METALES: FUERTE CORRELACIÓN")

# VISUALIZACIÓN ADICIONAL: Series temporales con anomalías destacadas
plt.figure(figsize=(16, 10))

# Soja con anomalías marcadas
plt.subplot(2, 1, 1)
plt.plot(df_clusters.index, df_clusters['SOJA'], 'b-', alpha=0.7, label='Precio Soja')
anomalias_soja = df_clusters[df_clusters['Es_Anomalia']]
plt.scatter(anomalias_soja.index, anomalias_soja['SOJA'], 
           c='red', marker='x', s=100, label='Días Anómalos', zorder=5)
plt.ylabel('Precio Soja (USD)')
plt.title('Precio de Soja con Anomalías Detectadas por DBSCAN')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Petróleo con anomalías marcadas
plt.subplot(2, 1, 2)
plt.plot(df_clusters.index, df_clusters['PETROLEO'], 'g-', alpha=0.7, label='Precio Petróleo')
anomalias_petroleo = df_clusters[df_clusters['Es_Anomalia']]
plt.scatter(anomalias_petroleo.index, anomalias_petroleo['PETROLEO'], 
           c='red', marker='x', s=100, label='Días Anómalos', zorder=5)
plt.ylabel('Precio Petróleo (USD)')
plt.xlabel('Fecha')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# RESUMEN EJECUTIVO
print("\n" + "="*70)
print("RESUMEN EJECUTIVO - DBSCAN COMMODITIES")
print("="*70)

print(f"""
 VENTAJAS DE DBSCAN EN ESTE ANÁLISIS:

• Detecta clusters de forma natural sin especificar número
• Identifica automáticamente outliers/anomalías ({n_ruido} días)
• Maneja clusters de forma y densidad arbitrarias
• Robustez ante ruido en datos financieros

 HALLAZGOS PRINCIPALES:

1. Se identificaron {n_clusters} patrones de mercado recurrentes
2. {n_ruido} días fueron marcados como anómalos ({n_ruido/len(clusters)*100:.1f}%)
3. Los clusters representan diferentes regímenes de correlación
4. Las anomalías corresponden a eventos de mercado inusuales

 APLICACIONES PRÁCTICAS:

• Detección temprana de cambios de régimen
• Identificación de oportunidades de arbitraje
• Gestión de riesgo mediante detección de outliers
• Estrategias basadas en patrones inter-mercados
""")

# Últimas anomalías detectadas
print("\nÚltimas 5 anomalías detectadas:")
print(anomalias[['SOJA', 'MAIZ', 'PETROLEO', 'ORO']].tail())

# Ventajas de DBSCAN en este contexto:
    # Detección de anomalías: Identifica días de mercado inusuales automáticamente
    # Clusters naturales: No fuerza número predefinido de grupos
    # Robustez: Maneja outliers mejor que K-Means
    # Formas arbitrarias: Detecta clusters de cualquier forma

 # Resultados esperados:
    # Clusters: 3-6 grupos naturales de comportamiento de mercado
    # Anomalías: 5-15% de días marcados como outliers
    # Patrones: Relaciones inter-mercados (agrícolas-energéticos-metales)

 # Casos de uso detectables:
    # Días de correlación extrema entre commodities
    # Eventos de desacoplamiento mercado agrícola-energético
    # Movimientos anómalos en metales preciosos
    # Cambios de régimen en relaciones inter-mercados
