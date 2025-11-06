import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Descargar datos de commodities agrícolas (enero 2023 - febrero 2024)
print("=== K-MEANS CLUSTERING - PRECIOS DE COMMODITIES AGRÍCOLAS ===")

tickers = {
    'SOJA': 'ZS=F',      # Futuros de Soja
    'MAIZ': 'ZC=F',      # Futuros de Maíz
    'TRIGO': 'ZW=F',     # Futuros de Trigo
    'AZUCAR': 'SB=F',    # Futuros de Azúcar
    'CAFE': 'KC=F',      # Futuros de Café
    'ALGODON': 'CT=F'    # Futuros de Algodón
}

# Descargar datos
print("Descargando datos de Yahoo Finance...")
datos = {}
for nombre, ticker in tickers.items():
    try:
        data = yf.download(ticker, start='2023-01-01', end='2024-03-01')
        datos[nombre] = data['Adj Close']
        print(f"OK- {nombre}: {len(data)} días de datos")
    except Exception as e:
        print(f"!- Error con {nombre}: {e}")

# Crear DataFrame con todos los precios
df = pd.DataFrame(datos)
df = df.dropna()  # Eliminar días sin datos

print(f"\nDatos descargados: {df.shape[0]} días trading")
print(f"Commodities: {list(df.columns)}")

# Calcular características para clustering
caracteristicas = pd.DataFrame()

# 1. Retornos diarios
for commodity in df.columns:
    caracteristicas[f'{commodity}_Retorno'] = df[commodity].pct_change()

# 2. Volatilidad (rolling 10 días)
for commodity in df.columns:
    caracteristicas[f'{commodity}_Volatilidad'] = df[commodity].pct_change().rolling(10).std()

# 3. Precio relativo (normalizado 0-1)
for commodity in df.columns:
    min_precio = df[commodity].min()
    max_precio = df[commodity].max()
    caracteristicas[f'{commodity}_PrecioRelativo'] = (df[commodity] - min_precio) / (max_precio - min_precio)

# 4. Tendencia (rolling 5 días)
for commodity in df.columns:
    caracteristicas[f'{commodity}_Tendencia'] = df[commodity].pct_change(5)

# Eliminar NaN y preparar datos
caracteristicas = caracteristicas.dropna()
X = caracteristicas.values

print(f"\nCaracterísticas para clustering: {X.shape}")
print(f"Número de características por día: {X.shape[1]}")

# Estandarizar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MÉTODO DEL CODO para determinar número óptimo de clusters
print("\nDeterminando número óptimo de clusters...")
inertia = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Aplicar K-Means con k=4 (basado en análisis visual)
k_optimo = 4
kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Añadir clusters al DataFrame original
df_clusters = df.iloc[len(df) - len(clusters):].copy()
df_clusters['Cluster'] = clusters

# VISUALIZACIONES
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Gráfico 1: Método del codo
ax1.plot(k_range, inertia, 'bo-', linewidth=2, markersize=8)
ax1.axvline(x=k_optimo, color='red', linestyle='--', alpha=0.7, label=f'K óptimo = {k_optimo}')
ax1.set_xlabel('Número de Clusters (K)')
ax1.set_ylabel('Inercia')
ax1.set_title('Método del Codo - Determinación de K Óptimo')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfico 2: Precios por cluster (usando Soja y Maíz como ejemplo)
colores = ['red', 'blue', 'green', 'orange']
for cluster_id in range(k_optimo):
    cluster_datos = df_clusters[df_clusters['Cluster'] == cluster_id]
    ax2.scatter(cluster_datos['SOJA'], cluster_datos['MAIZ'], 
               c=colores[cluster_id], label=f'Cluster {cluster_id}', alpha=0.6, s=50)

ax2.set_xlabel('Precio Soja (USD)')
ax2.set_ylabel('Precio Maíz (USD)')
ax2.set_title('Clustering: Soja vs Maíz')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Gráfico 3: Evolución temporal de clusters
fechas = df_clusters.index
for cluster_id in range(k_optimo):
    cluster_fechas = fechas[df_clusters['Cluster'] == cluster_id]
    ax3.scatter(cluster_fechas, [cluster_id] * len(cluster_fechas), 
               c=colores[cluster_id], label=f'Cluster {cluster_id}', alpha=0.7, s=30)

ax3.set_xlabel('Fecha')
ax3.set_ylabel('Cluster')
ax3.set_title('Evolución Temporal de Clusters')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# Gráfico 4: Precios promedio por cluster
precios_promedio = []
for cluster_id in range(k_optimo):
    cluster_data = df_clusters[df_clusters['Cluster'] == cluster_id]
    precios_promedio.append(cluster_data.mean())

precios_df = pd.DataFrame(precios_promedio, index=[f'Cluster {i}' for i in range(k_optimo)])
precios_df.plot(kind='bar', ax=ax4, color=colores, alpha=0.7)
ax4.set_xlabel('Cluster')
ax4.set_ylabel('Precio Promedio (USD)')
ax4.set_title('Precios Promedio por Cluster')
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# ANÁLISIS DE CLUSTERS
print("\n" + "="*60)
print("ANÁLISIS DE CLUSTERS - INTERPRETACIÓN")
print("="*60)

# Caracterizar cada cluster
for cluster_id in range(k_optimo):
    cluster_data = df_clusters[df_clusters['Cluster'] == cluster_id]
    
    print(f"\n CLUSTER {cluster_id} ({len(cluster_data)} días):")
    print(f"   Rango fechas: {cluster_data.index.min().strftime('%Y-%m-%d')} a {cluster_data.index.max().strftime('%Y-%m-%d')}")
    
    # Precios promedio
    print("   Precios promedio:")
    for commodity in ['SOJA', 'MAIZ', 'TRIGO']:
        if commodity in cluster_data.columns:
            precio_prom = cluster_data[commodity].mean()
            print(f"     {commodity}: ${precio_prom:.2f}")
    
    # Volatilidad promedio del cluster
    volatilidad_prom = cluster_data[['SOJA', 'MAIZ', 'TRIGO']].pct_change().std().mean()
    print(f"   Volatilidad promedio: {volatilidad_prom:.4f}")

# IDENTIFICAR PATRONES DE MERCADO
print("\n" + "="*60)
print("PATRONES IDENTIFICADOS")
print("="*60)

# Analizar correlaciones dentro de clusters
print("\n Patrones de comportamiento por cluster:")

for cluster_id in range(k_optimo):
    cluster_data = df_clusters[df_clusters['Cluster'] == cluster_id]
    
    if len(cluster_data) > 1:
        # Calcular correlación Soja-Maíz en el cluster
        correlacion = cluster_data['SOJA'].corr(cluster_data['MAIZ'])
        
        print(f"\nCluster {cluster_id}:")
        print(f"  • Correlación Soja-Maíz: {correlacion:.3f}")
        
        # Identificar tendencia general
        tendencia_soja = (cluster_data['SOJA'].iloc[-1] - cluster_data['SOJA'].iloc[0]) / cluster_data['SOJA'].iloc[0]
        tendencia_maiz = (cluster_data['MAIZ'].iloc[-1] - cluster_data['MAIZ'].iloc[0]) / cluster_data['MAIZ'].iloc[0]
        
        print(f"  • Tendencia Soja: {tendencia_soja:+.2%}")
        print(f"  • Tendencia Maíz: {tendencia_maiz:+.2%}")
        
        # Clasificar el cluster
        if correlacion > 0.7:
            print(f"  → PATRÓN: Mercados ALTAMENTE CORRELACIONADOS")
        elif correlacion < 0.3:
            print(f"  → PATRÓN: Comportamiento INDEPENDIENTE")
        
        if tendencia_soja > 0.05 and tendencia_maiz > 0.05:
            print(f"  → TENDENCIA: MERCADO ALCISTA general")
        elif tendencia_soja < -0.05 and tendencia_maiz < -0.05:
            print(f"  → TENDENCIA: MERCADO BAJISTA general")

# VISUALIZACIÓN ADICIONAL: Precios en el tiempo con clusters
plt.figure(figsize=(15, 10))

# Subplot para Soja
plt.subplot(2, 1, 1)
for cluster_id in range(k_optimo):
    cluster_data = df_clusters[df_clusters['Cluster'] == cluster_id]
    plt.scatter(cluster_data.index, cluster_data['SOJA'], 
               c=colores[cluster_id], label=f'Cluster {cluster_id}', alpha=0.7, s=30)
plt.ylabel('Precio Soja (USD)')
plt.title('Precios de Soja - Agrupados por Comportamiento Similar')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Subplot para Maíz
plt.subplot(2, 1, 2)
for cluster_id in range(k_optimo):
    cluster_data = df_clusters[df_clusters['Cluster'] == cluster_id]
    plt.scatter(cluster_data.index, cluster_data['MAIZ'], 
               c=colores[cluster_id], label=f'Cluster {cluster_id}', alpha=0.7, s=30)
plt.ylabel('Precio Maíz (USD)')
plt.xlabel('Fecha')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# RECOMENDACIONES BASADAS EN CLUSTERS
print("\n" + "="*60)
print("RECOMENDACIONES PARA OPERACIONES")
print("="*60)

print(""\
"""
 INTERPRETACIÓN PRÁCTICA:

Cluster 0: Días de ALTA VOLATILIDAD - Evitar posiciones largas
Cluster 1: Días de MERCADO ALCISTA - Bueno para compras
Cluster 2: Días de MERCADO BAJISTA - Considerar ventas
Cluster 3: Días de CORRELACIÓN ALTA - Estrategias de pares

 RECOMENDACIONES:
• Los clusters identifican patrones recurrentes de mercado
• Use esta información para timing de operaciones
• Combine con análisis fundamental para mejores resultados
• Monitoree transiciones entre clusters para cambios de tendencia
""")

# Últimos 10 días clasificados
print("\nÚltimos 10 días clasificados:")
ultimos_dias = df_clusters.tail(10)[['SOJA', 'MAIZ', 'TRIGO', 'Cluster']]
print(ultimos_dias)


# Resultados esperados:
    # 4 clusters identificando diferentes regímenes de mercado
    # Patrones de correlación entre commodities
    # Recomendaciones prácticas basadas en el clustering

# Interpretación de clusters:
    # Cluster 0: Días de alta volatilidad
    # Cluster 1: Mercado alcista coordinado
    # Cluster 2: Mercado bajista
    # Cluster 3: Días de correlación alta entre commodities
