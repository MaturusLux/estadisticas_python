import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from fcmeans import FCM
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Configuración
plt.style.use('seaborn-v0_8')
np.random.seed(42)

# 1. Simulación de Datos de Clientes/Proveedores

def simular_datos_agroempresa(n_clientes=300):
    """Simular datos de clientes que también son proveedores"""
    
    np.random.seed(42)
    
    datos = {}
    
    # Información base del cliente
    datos['id_cliente'] = range(1, n_clientes + 1)
    datos['antiguedad_meses'] = np.random.gamma(20, 3, n_clientes).astype(int)
    datos['tamaño_hectareas'] = np.random.lognormal(5, 1, n_clientes)
    
    # COMPRAS POR RUBRO (comportamiento como cliente)
    # 1. Agroquímicos (algunos compran mucho, otros poco)
    datos['compras_agroquimicos'] = np.random.weibull(1.5, n_clientes) * 10000
    
    # 2. Semillas (estacional, algunos especializados)
    datos['compras_semillas'] = np.random.normal(5000, 2000, n_clientes)
    
    # 3. Combustible & Lubricantes (relacionado con tamaño y maquinaria)
    datos['compras_combustible'] = datos['tamaño_hectareas'] * np.random.normal(50, 15, n_clientes)
    
    # VENTAS A LA EMPRESA (comportamiento como proveedor)
    datos['ventas_cosecha_soja'] = np.random.exponential(100, n_clientes) * datos['tamaño_hectareas']
    datos['ventas_cosecha_maiz'] = np.random.exponential(80, n_clientes) * datos['tamaño_hectareas']
    datos['ventas_cosecha_trigo'] = np.random.exponential(60, n_clientes) * datos['tamaño_hectareas']
    
    # Características adicionales
    datos['frecuencia_compra'] = np.random.poisson(8, n_clientes)
    datos['credito_utilizado'] = np.random.uniform(0, 50000, n_clientes)
    datos['distancia_km'] = np.random.exponential(50, n_clientes)
    
    df = pd.DataFrame(datos)
    
    # Asegurar valores positivos
    columnas_monetarias = ['compras_agroquimicos', 'compras_semillas', 'compras_combustible',
                          'ventas_cosecha_soja', 'ventas_cosecha_maiz', 'ventas_cosecha_trigo',
                          'credito_utilizado']
    
    for col in columnas_monetarias:
        df[col] = np.abs(df[col])
    
    return df

# Generar datos
df_clientes = simular_datos_agroempresa(300)
print("📊 DATOS DE CLIENTES/PROVEEDORES")
print(f"Dimensiones: {df_clientes.shape}")
print("\nPrimeras filas:")
print(df_clientes.head().round(2))

#  2. Preparación de Características para Fuzzy C-Means
def preparar_caracteristicas_fuzzy(df):
    """Preparar características para el clustering difuso"""
    
    # Crear variables de participación por rubro
    df_analisis = df.copy()
    
    # 1. Participación en COMPRAS (como cliente)
    total_compras = (df['compras_agroquimicos'] + df['compras_semillas'] + df['compras_combustible'])
    
    df_analisis['participacion_agroquimicos'] = df['compras_agroquimicos'] / total_compras
    df_analisis['participacion_semillas'] = df['compras_semillas'] / total_compras
    df_analisis['participacion_combustible'] = df['compras_combustible'] / total_compras
    
    # 2. Participación en VENTAS (como proveedor)
    total_ventas = (df['ventas_cosecha_soja'] + df['ventas_cosecha_maiz'] + df['ventas_cosecha_trigo'])
    
    # Evitar división por cero
    total_ventas = np.where(total_ventas == 0, 1, total_ventas)
    
    df_analisis['participacion_ventas_soja'] = df['ventas_cosecha_soja'] / total_ventas
    df_analisis['participacion_ventas_maiz'] = df['ventas_cosecha_maiz'] / total_ventas
    df_analisis['participacion_ventas_trigo'] = df['ventas_cosecha_trigo'] / total_ventas
    
    # 3. Balance cliente-proveedor
    df_analisis['ratio_cliente_proveedor'] = total_compras / (total_ventas + 1)  # +1 para evitar div/0
    
    # 4. Características de valor
    df_analisis['valor_total_transacciones'] = total_compras + total_ventas
    df_analisis['frecuencia_normalizada'] = df['frecuencia_compra'] / df['frecuencia_compra'].max()
    
    # Seleccionar características para clustering
    caracteristicas = [
        'participacion_agroquimicos', 'participacion_semillas', 'participacion_combustible',
        'participacion_ventas_soja', 'participacion_ventas_maiz', 'participacion_ventas_trigo',
        'ratio_cliente_proveedor', 'valor_total_transacciones', 'frecuencia_normalizada',
        'tamaño_hectareas', 'antiguedad_meses'
    ]
    
    X = df_analisis[caracteristicas]
    
    # Estandarizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, df_analisis, caracteristicas

# Preparar datos
X_scaled, df_analisis, features = preparar_caracteristicas_fuzzy(df_clientes)
print(f"🎯 CARACTERÍSTICAS PARA FUZZY C-MEANS: {len(features)} variables")
print(features)

# 3. Aplicación de Fuzzy C-Means
def aplicar_fuzzy_c_means(X, n_clusters=4):
    """Aplicar Fuzzy C-Means y analizar resultados"""
    
    # Aplicar Fuzzy C-Means
    fcm = FCM(n_clusters=n_clusters, random_state=42)
    fcm.fit(X)
    
    # Obtener pertenencias difusas y clusters duros
    pertenencias = fcm.soft_predict(X)  # Matriz de pertenencias [n_muestras x n_clusters]
    clusters = fcm.predict(X)           # Cluster más probable para cada cliente
    
    # Métricas de calidad
    from sklearn.metrics import silhouette_score
    sil_score = silhouette_score(X, clusters)
    
    print("📊 RESULTADOS FUZZY C-MEANS")
    print("="*50)
    print(f"Número de clusters: {n_clusters}")
    print(f"Silhouette Score: {sil_score:.4f}")
    print(f"Función objetivo final: {fcm.objective_function:.4f}")
    print(f"Número de iteraciones: {fcm.n_iter}")
    
    return fcm, pertenencias, clusters, sil_score

# Aplicar Fuzzy C-Means
fcm, pertenencias, clusters, sil_score = aplicar_fuzzy_c_means(X_scaled, n_clusters=4)

# Añadir resultados al DataFrame
df_analisis['cluster'] = clusters
for i in range(4):
    df_analisis[f'pertenencia_cluster_{i}'] = pertenencias[:, i]

print("\n📈 DISTRIBUCIÓN DE CLIENTES POR CLUSTER:")
print(df_analisis['cluster'].value_counts().sort_index())

# 4. Visualización de Resultados
def visualizar_resultados_fuzzy(df_analisis, pertenencias, clusters):
    """Visualizar resultados del clustering difuso"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Distribución de pertenencias
    for i in range(4):
        axes[0, 0].hist(pertenencias[:, i], bins=30, alpha=0.7, 
                       label=f'Cluster {i}', density=True)
    axes[0, 0].set_title('Distribución de Grados de Pertenencia')
    axes[0, 0].set_xlabel('Grado de Pertenencia')
    axes[0, 0].set_ylabel('Densidad')
    axes[0, 0].legend()
    
    # 2. Composición de clusters (participación en compras)
    participacion_compras = df_analisis.groupby('cluster')[
        ['participacion_agroquimicos', 'participacion_semillas', 'participacion_combustible']
    ].mean()
    
    participacion_compras.plot(kind='bar', ax=axes[0, 1], 
                              color=['#2E8B57', '#FFD700', '#FF6347'])
    axes[0, 1].set_title('Participación Promedio en COMPRAS por Cluster')
    axes[0, 1].set_ylabel('Participación Promedia')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Composición de clusters (participación en ventas)
    participacion_ventas = df_analisis.groupby('cluster')[
        ['participacion_ventas_soja', 'participacion_ventas_maiz', 'participacion_ventas_trigo']
    ].mean()
    
    participacion_ventas.plot(kind='bar', ax=axes[0, 2], 
                             color=['#8B4513', '#FFA500', '#F0E68C'])
    axes[0, 2].set_title('Participación Promedio en VENTAS por Cluster')
    axes[0, 2].set_ylabel('Participación Promedia')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. Mapa de calor de pertenencias
    # Ordenar por cluster principal y grado de pertenencia
    df_ordenado = df_analisis.sort_values(['cluster'] + 
                                        [f'pertenencia_cluster_{i}' for i in range(4)], 
                                        ascending=[True] + [False]*4)
    
    im = axes[1, 0].imshow(pertenencias[df_ordenado.index], aspect='auto', 
                          cmap='YlOrRd', interpolation='nearest')
    axes[1, 0].set_title('Matriz de Pertenencias Difusas\n(Ordenado por Cluster)')
    axes[1, 0].set_xlabel('Cluster')
    axes[1, 0].set_ylabel('Cliente (Ordenado)')
    axes[1, 0].set_xticks(range(4))
    axes[1, 0].set_xticklabels([f'C{i}' for i in range(4)])
    plt.colorbar(im, ax=axes[1, 0], label='Grado de Pertenencia')
    
    # 5. Relación Cliente-Proveedor por Cluster
    scatter = axes[1, 1].scatter(df_analisis['participacion_agroquimicos'], 
                               df_analisis['participacion_ventas_soja'],
                               c=clusters, cmap='viridis', 
                               alpha=0.6, s=50)
    axes[1, 1].set_xlabel('Participación Agroquímicos (Cliente)')
    axes[1, 1].set_ylabel('Participación Soja (Proveedor)')
    axes[1, 1].set_title('Relación Cliente-Proveedor por Cluster')
    plt.colorbar(scatter, ax=axes[1, 1], label='Cluster')
    
    # 6. Valor vs Frecuencia
    scatter = axes[1, 2].scatter(df_analisis['valor_total_transacciones'], 
                               df_analisis['frecuencia_normalizada'],
                               c=clusters, cmap='viridis', 
                               alpha=0.6, s=50)
    axes[1, 2].set_xlabel('Valor Total de Transacciones')
    axes[1, 2].set_ylabel('Frecuencia Normalizada')
    axes[1, 2].set_title('Valor vs Frecuencia por Cluster')
    plt.colorbar(scatter, ax=axes[1, 2], label='Cluster')
    
    plt.tight_layout()
    plt.show()

# Visualizar resultados
visualizar_resultados_fuzzy(df_analisis, pertenencias, clusters)

# 5. Interpretación de Clusters para Negocio
def interpretar_clusters_negocio(df_analisis):
    """Interpretar los clusters desde la perspectiva de negocio"""
    
    print("🎯 INTERPRETACIÓN DE CLUSTERS - AGROSUPPLY S.A.")
    print("="*60)
    
    # Analizar cada cluster
    for cluster_id in range(4):
        cluster_data = df_analisis[df_analisis['cluster'] == cluster_id]
        
        print(f"\n📊 CLUSTER {cluster_id} - {len(cluster_data)} clientes/proveedores")
        print("-" * 50)
        
        # Características promedio
        print("📈 CARACTERÍSTICAS PROMEDIO:")
        print(f"  • Tamaño promedio: {cluster_data['tamaño_hectareas'].mean():.1f} ha")
        print(f"  • Antigüedad: {cluster_data['antiguedad_meses'].mean():.1f} meses")
        print(f"  • Valor total transacciones: ${cluster_data['valor_total_transacciones'].mean():.0f}")
        print(f"  • Ratio Cliente/Proveedor: {cluster_data['ratio_cliente_proveedor'].mean():.2f}")
        
        # Comportamiento en COMPRAS
        print("\n🛒 COMPORTAMIENTO COMO CLIENTE:")
        print(f"  • Agroquímicos: {cluster_data['participacion_agroquimicos'].mean()*100:.1f}%")
        print(f"  • Semillas: {cluster_data['participacion_semillas'].mean()*100:.1f}%")
        print(f"  • Combustible: {cluster_data['participacion_combustible'].mean()*100:.1f}%")
        
        # Comportamiento en VENTAS
        print("\n🌾 COMPORTAMIENTO COMO PROVEEDOR:")
        print(f"  • Soja: {cluster_data['participacion_ventas_soja'].mean()*100:.1f}%")
        print(f"  • Maíz: {cluster_data['participacion_ventas_maiz'].mean()*100:.1f}%")
        print(f"  • Trigo: {cluster_data['participacion_ventas_trigo'].mean()*100:.1f}%")
        
        # Asignar nombre descriptivo al cluster
        nombre_cluster = asignar_nombre_cluster(cluster_data)
        print(f"\n🏷️  PERFIL: {nombre_cluster}")
        
        # Recomendaciones de negocio
        print("💡 RECOMENDACIONES:")
        recomendaciones = generar_recomendaciones(cluster_id, cluster_data)
        for rec in recomendaciones:
            print(f"  • {rec}")

def asignar_nombre_cluster(cluster_data):
    """Asignar nombre descriptivo basado en características del cluster"""
    
    avg_agroquimicos = cluster_data['participacion_agroquimicos'].mean()
    avg_semillas = cluster_data['participacion_semillas'].mean()
    avg_combustible = cluster_data['participacion_combustible'].mean()
    avg_ratio = cluster_data['ratio_cliente_proveedor'].mean()
    avg_valor = cluster_data['valor_total_transacciones'].mean()
    
    if avg_ratio > 2.0:
        return "CLIENTES ESTRATÉGICOS - Alto consumo"
    elif avg_ratio < 0.5:
        return "PROVEEDORES PRINCIPALES - Alta venta de cosecha"
    elif avg_agroquimicos > 0.5:
        return "ESPECIALISTAS EN AGROQUÍMICOS"
    elif avg_semillas > 0.4:
        return "CLIENTES DE SEMILLAS"
    elif avg_combustible > 0.4:
        return "GRANDES CONSUMIDORES DE COMBUSTIBLE"
    else:
        return "CLIENTES BALANCEADOS"

def generar_recomendaciones(cluster_id, cluster_data):
    """Generar recomendaciones específicas por cluster"""
    
    recomendaciones = []
    avg_ratio = cluster_data['ratio_cliente_proveedor'].mean()
    avg_valor = cluster_data['valor_total_transacciones'].mean()
    
    if cluster_id == 0:  # CLIENTES ESTRATÉGICOS
        recomendaciones.extend([
            "Programas de fidelización premium",
            "Descuentos por volumen en agroquímicos",
            "Asesoramiento técnico personalizado"
        ])
    
    elif cluster_id == 1:  # PROVEEDORES PRINCIPALES
        recomendaciones.extend([
            "Contratos de compra preferencial",
            "Financiamiento para insumos de siembra",
            "Programas de precios diferenciales"
        ])
    
    elif cluster_id == 2:  # ESPECIALISTAS EN AGROQUÍMICOS
        recomendaciones.extend([
            "Promociones de semillas complementarias",
            "Paquetes integrados semilla-agroquímico",
            "Capacitación en manejo integrado"
        ])
    
    elif cluster_id == 3:  # CLIENTES BALANCEADOS
        recomendaciones.extend([
            "Programas de cross-selling",
            "Ofertas personalizadas por cultivo",
            "Seguimiento intensivo de relación"
        ])
    
    # Recomendaciones generales basadas en métricas
    if avg_valor > df_analisis['valor_total_transacciones'].quantile(0.75):
        recomendaciones.append("Gestión de cuenta por ejecutivo especializado")
    
    if cluster_data['frecuencia_normalizada'].mean() > 0.7:
        recomendaciones.append("Programas de compra recurrente con beneficios")
    
    return recomendaciones

# Interpretar clusters
interpretar_clusters_negocio(df_analisis)

# 6. Análisis de Pertenencia Difusa por Cliente
def analisis_individual_clientes(df_analisis, n_ejemplos=5):
    """Analizar casos individuales mostrando la naturaleza difusa"""
    
    print("\n🔍 ANÁLISIS INDIVIDUAL DE CLIENTES - NATURALEZA DIFUSA")
    print("="*60)
    
    # Encontrar clientes con pertenencias mixtas (más interesantes)
    df_analisis['max_pertenencia'] = df_analisis[[f'pertenencia_cluster_{i}' for i in range(4)]].max(axis=1)
    df_analisis['entropia_pertenencias'] = -np.sum(
        pertenencias * np.log(pertenencias + 1e-10), axis=1
    )
    
    # Clientes con alta entropía (pertenencias mixtas)
    clientes_mixtos = df_analisis.nlargest(n_ejemplos, 'entropia_pertenencias')
    
    for idx, (_, cliente) in enumerate(clientes_mixtos.iterrows()):
        print(f"\n👤 CLIENTE {int(cliente['id_cliente'])} - PERTENENCIAS MIXTAS:")
        print(f"   Cluster principal: {int(cliente['cluster'])}")
        print(f"   Entropía: {cliente['entropia_pertenencias']:.3f}")
        
        print("   Grados de pertenencia:")
        for i in range(4):
            print(f"     Cluster {i}: {cliente[f'pertenencia_cluster_{i}']:.3f}")
        
        print("   Comportamiento comercial:")
        print(f"     Compra Agroq: {cliente['participacion_agroquimicos']:.1%}")
        print(f"     Compra Semillas: {cliente['participacion_semillas']:.1%}")
        print(f"     Venta Soja: {cliente['participacion_ventas_soja']:.1%}")
        print(f"     Ratio C/P: {cliente['ratio_cliente_proveedor']:.2f}")

# Analizar casos individuales
analisis_individual_clientes(df_analisis)

# 7. Aplicaciones Prácticas para la Empresa
def aplicaciones_practicas_empresa(df_analisis):
    """Mostrar aplicaciones prácticas del análisis difuso"""
    
    print("\n🚀 APLICACIONES PRÁCTICAS PARA AGROSUPPLY S.A.")
    print("="*60)
    
    # 1. Segmentación para marketing
    print("\n📧 SEGMENTACIÓN PARA CAMPAÑAS DE MARKETING:")
    for cluster_id in range(4):
        n_clientes = (df_analisis['cluster'] == cluster_id).sum()
        perfil = asignar_nombre_cluster(df_analisis[df_analisis['cluster'] == cluster_id])
        
        print(f"  Cluster {cluster_id} ({perfil}):")
        print(f"    • {n_clientes} clientes")
        
        # Recomendación de producto principal
        participaciones = df_analisis[df_analisis['cluster'] == cluster_id][
            ['participacion_agroquimicos', 'participacion_semillas', 'participacion_combustible']
        ].mean()
        
        producto_principal = participaciones.idxmax().replace('participacion_', '')
        print(f"    • Producto principal: {producto_principal}")
        print(f"    • Estrategia: Campañas específicas de {producto_principal.replace('_', ' ')}")
    
    # 2. Gestión de relaciones
    print("\n🤝 GESTIÓN DE RELACIONES:")
    print("  Clientes con alta pertenencia a múltiples clusters → Asignar ejecutivos senior")
    print("  Clientes con baja entropía → Estrategias específicas por cluster")
    print("  Ratio Cliente/Proveedor balanceado → Programas de lealtad integrados")
    
    # 3. Optimización de inventario
    print("\n📦 OPTIMIZACIÓN DE INVENTARIO POR REGIÓN:")
    # Agrupar por características geográficas (simulado)
    print("  Usar grados de pertenencia para predecir demanda por región")
    print("  Asignar stock basado en composición de clusters por sucursal")

# Mostrar aplicaciones prácticas
aplicaciones_practicas_empresa(df_analisis)

