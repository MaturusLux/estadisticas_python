#  Ejemplo Práctico Comparativo
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from sklearn.datasets import make_blobs, make_circles, make_moons, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
import pandas as pd
import time

# Configuración
plt.style.use('seaborn-v0_8')
np.random.seed(42)

#1. Función de comparación visual
def comparar_tsne_umap_visual(datasets):
    """Comparación visual entre t-SNE y UMAP en diferentes datasets"""
    
    fig, axes = plt.subplots(len(datasets), 3, figsize=(15, 5*len(datasets)))
    
    if len(datasets) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (X, y, nombre) in enumerate(datasets):
        # Estandarizar
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        
        # Visualizar datos originales (si son 2D/3D)
        if X.shape[1] <= 3:
            if X.shape[1] == 2:
                axes[idx, 0].scatter(X_std[:, 0], X_std[:, 1], c=y, cmap='viridis', alpha=0.7)
            else:
                # Para 3D, usar proyección 2D
                axes[idx, 0].scatter(X_std[:, 0], X_std[:, 1], c=y, cmap='viridis', alpha=0.7)
        else:
            # Para alta dimensión, mostrar histograma de una característica
            axes[idx, 0].hist(X_std[:, 0], alpha=0.7, bins=30)
            axes[idx, 0].set_title(f'{nombre}\nDistribución Feature 1')
        
        axes[idx, 0].set_title(f'{nombre}\nOriginal')
        
        # Aplicar t-SNE
        start_time = time.time()
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_std)
        tiempo_tsne = time.time() - start_time
        
        # Aplicar UMAP
        start_time = time.time()
        reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
        X_umap = reducer.fit_transform(X_std)
        tiempo_umap = time.time() - start_time
        
        # Visualizar t-SNE
        scatter_tsne = axes[idx, 1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
        axes[idx, 1].set_title(f't-SNE\nTiempo: {tiempo_tsne:.2f}s')
        
        # Visualizar UMAP
        scatter_umap = axes[idx, 2].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis', alpha=0.7)
        axes[idx, 2].set_title(f'UMAP\nTiempo: {tiempo_umap:.2f}s')
        
        # Calcular métricas
        if len(np.unique(y)) > 1:
            sil_tsne = silhouette_score(X_tsne, y)
            sil_umap = silhouette_score(X_umap, y)
            
            axes[idx, 1].set_xlabel(f'Silhouette: {sil_tsne:.3f}')
            axes[idx, 2].set_xlabel(f'Silhouette: {sil_umap:.3f}')
    
    plt.tight_layout()
    plt.show()

    # 2. Crear datasets de prueba
    def crear_datasets_comparativos():
    """Crear diversos datasets para comparar t-SNE y UMAP"""
    
    datasets = []
    
    # 1. Clusters bien separados (fácil)
    X1, y1 = make_blobs(n_samples=300, centers=4, n_features=10, random_state=42)
    datasets.append((X1, y1, "Clusters Bien Separados"))
    
    # 2. Anillos concéntricos (no lineal)
    X2, y2 = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)
    # Añadir dimensiones extra con ruido
    X2_extra = np.column_stack([X2, np.random.normal(0, 0.5, (300, 8))])
    datasets.append((X2_extra, y2, "Anillos + Ruido 8D"))
    
    # 3. Lunas (no lineal)
    X3, y3 = make_moons(n_samples=300, noise=0.1, random_state=42)
    X3_extra = np.column_stack([X3, np.random.normal(0, 0.5, (300, 8))])
    datasets.append((X3_extra, y3, "Lunas + Ruido 8D"))
    
    # 4. Datos de alta dimensión
    X4, y4 = make_blobs(n_samples=300, centers=5, n_features=50, random_state=42)
    datasets.append((X4, y4, "Alta Dimensión (50D)"))
    
    return datasets

# Crear y comparar datasets
datasets_comparativos = crear_datasets_comparativos()
comparar_tsne_umap_visual(datasets_comparativos)

# 3. Comparación cuantitativa detallada
def comparacion_cuantitativa_tsne_umap():
    """Comparación cuantitativa exhaustiva entre t-SNE y UMAP"""
    
    # Dataset más complejo
    X, y = make_blobs(n_samples=500, centers=6, n_features=20, random_state=42)
    X_std = StandardScaler().fit_transform(X)
    
    resultados = []
    
    # Probar diferentes parámetros
    perplexities = [5, 15, 30, 50]
    n_neighbors_list = [5, 15, 30, 50]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # t-SNE con diferentes perplexities
    for i, perplexity in enumerate(perplexities):
        start_time = time.time()
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_tsne = tsne.fit_transform(X_std)
        tiempo_tsne = time.time() - start_time
        
        sil_tsne = silhouette_score(X_tsne, y)
        
        axes[0, i].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
        axes[0, i].set_title(f't-SNE Perplexity={perplexity}\nSil: {sil_tsne:.3f}, Time: {tiempo_tsne:.2f}s')
        
        resultados.append({
            'metodo': 't-SNE',
            'parametro': f'perplexity={perplexity}',
            'silhouette': sil_tsne,
            'tiempo': tiempo_tsne
        })
    
    # UMAP con diferentes n_neighbors
    for i, n_neighbors in enumerate(n_neighbors_list):
        start_time = time.time()
        reducer = umap.UMAP(random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
        X_umap = reducer.fit_transform(X_std)
        tiempo_umap = time.time() - start_time
        
        sil_umap = silhouette_score(X_umap, y)
        
        axes[1, i].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis', alpha=0.7)
        axes[1, i].set_title(f'UMAP n_neighbors={n_neighbors}\nSil: {sil_umap:.3f}, Time: {tiempo_umap:.2f}s')
        
        resultados.append({
            'metodo': 'UMAP',
            'parametro': f'n_neighbors={n_neighbors}',
            'silhouette': sil_umap,
            'tiempo': tiempo_umap
        })
    
    plt.tight_layout()
    plt.show()
    
    # DataFrame de resultados
    df_resultados = pd.DataFrame(resultados)
    print(" COMPARACIÓN CUANTITATIVA t-SNE vs UMAP")
    print("="*50)
    print(df_resultados.round(4))
    
    return df_resultados

# Ejecutar comparación cuantitativa
df_comparacion = comparacion_cuantitativa_tsne_umap()

# 4. Caso real: Dígitos MNIST
def comparacion_mnist_tsne_umap():
    """Comparación en dataset real MNIST"""
    
    # Cargar dígitos MNIST
    digits = load_digits()
    X, y = digits.data, digits.target
    
    print(f" Dataset MNIST: {X.shape[0]} muestras, {X.shape[1]} características")
    
    # Estandarizar
    X_std = StandardScaler().fit_transform(X)
    
    # t-SNE
    print("Aplicando t-SNE...")
    start_time = time.time()
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_std)
    tiempo_tsne = time.time() - start_time
    
    # UMAP
    print("Aplicando UMAP...")
    start_time = time.time()
    reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
    X_umap = reducer.fit_transform(X_std)
    tiempo_umap = time.time() - start_time
    
    # Visualizar comparación
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # t-SNE
    scatter_tsne = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7)
    axes[0].set_title(f't-SNE MNIST\nTiempo: {tiempo_tsne:.2f}s')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    
    # UMAP
    scatter_umap = axes[1].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', alpha=0.7)
    axes[1].set_title(f'UMAP MNIST\nTiempo: {tiempo_umap:.2f}s')
    axes[1].set_xlabel('UMAP 1')
    axes[1].set_ylabel('UMAP 2')
    
    # Comparación lado a lado con mismos ejes
    for i in range(10):
        mask = y == i
        axes[2].scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       label=f'{i} (t-SNE)', alpha=0.6, s=30, marker='o')
        axes[2].scatter(X_umap[mask, 0], X_umap[mask, 1], 
                       label=f'{i} (UMAP)', alpha=0.6, s=30, marker='s')
    
    axes[2].set_title('Comparación Superpuesta')
    axes[2].set_xlabel('Componente 1')
    axes[2].set_ylabel('Componente 2')
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.colorbar(scatter_tsne, ax=axes[0], label='Dígito')
    plt.colorbar(scatter_umap, ax=axes[1], label='Dígito')
    
    plt.tight_layout()
    plt.show()
    
    # Métricas
    sil_tsne = silhouette_score(X_tsne, y)
    sil_umap = silhouette_score(X_umap, y)
    
    print(f"\n📈 MÉTRICAS MNIST:")
    print(f"t-SNE - Silhouette: {sil_tsne:.4f}, Tiempo: {tiempo_tsne:.2f}s")
    print(f"UMAP  - Silhouette: {sil_umap:.4f}, Tiempo: {tiempo_umap:.2f}s")
    print(f"Velocidad UMAP vs t-SNE: {tiempo_tsne/tiempo_umap:.1f}x más rápido")
    
    return X_tsne, X_umap, y

# Ejecutar comparación MNIST
X_tsne_mnist, X_umap_mnist, y_mnist = comparacion_mnist_tsne_umap()

# 5. UMAP en Selección de Personal (continuación del ejemplo anterior)
def umap_seleccion_personal(df_candidatos):
    """Aplicar UMAP al caso de selección de personal"""
    
    # Preparar datos
    X = df_candidatos.drop('perfil_oculto', axis=1)
    y = df_candidatos['perfil_oculto']
    
    # Estandarizar
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    # Aplicar UMAP
    reducer = umap.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=15,      # Similar a perplexity en t-SNE
        min_dist=0.1,        # Distancia mínima entre puntos
        metric='euclidean'   # Métrica de distancia
    )

    		# UMAP óptimo para la mayoría de casos
			# umap_optimo = umap.UMAP(
			#     n_components=2,
			#     n_neighbors=15,      # Balance local/global
			#     min_dist=0.1,        # Espaciado entre puntos
			#     metric='euclidean',  # Tipo de distancia
			#     random_state=42      # Reproducibilidad
			# )
    
    X_umap = reducer.fit_transform(X_std)
    
    # Visualizar resultados
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # t-SNE para comparación
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_std)
    
    		# t-SNE para comparación
			# tsne_optimo = TSNE(
			#     n_components=2,
			#     perplexity=30,       # Similar a n_neighbors
			#     random_state=42,
			#     n_iter=1000
			# )


    # UMAP
    scatter_umap = axes[0].scatter(X_umap[:, 0], X_umap[:, 1], 
                                 c=pd.factorize(y)[0], cmap='viridis', 
                                 alpha=0.7, s=60)
    axes[0].set_title('UMAP - Selección de Personal\n(Preserva mejor estructura global)')
    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')
    
    # t-SNE
    scatter_tsne = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                 c=pd.factorize(y)[0], cmap='viridis', 
                                 alpha=0.7, s=60)
    axes[1].set_title('t-SNE - Selección de Personal\n(Enfocado en estructura local)')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    
    # Añadir métricas
    sil_umap = silhouette_score(X_umap, pd.factorize(y)[0])
    sil_tsne = silhouette_score(X_tsne, pd.factorize(y)[0])
    
    axes[0].text(0.02, 0.98, f'Silhouette: {sil_umap:.3f}', 
                transform=axes[0].transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    axes[1].text(0.02, 0.98, f'Silhouette: {sil_tsne:.3f}', 
                transform=axes[1].transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Análisis de clusters con UMAP
    from sklearn.cluster import KMeans
    
    # Encontrar clusters óptimos
    kmeans_umap = KMeans(n_clusters=4, random_state=42)
    clusters_umap = kmeans_umap.fit_predict(X_umap)
    
    # Interpretación para RRHH
    df_analisis = df_candidatos.copy()
    df_analisis['umap_1'] = X_umap[:, 0]
    df_analisis['umap_2'] = X_umap[:, 1]
    df_analisis['cluster_umap'] = clusters_umap
    
    print(" INTERPRETACIÓN UMAP PARA RRHH:")
    print("="*50)
    
    # Analizar cada cluster
    for cluster_id in range(4):
        mask = df_analisis['cluster_umap'] == cluster_id
        cluster_data = df_analisis[mask]
        
        print(f"\n📊 Cluster {cluster_id} ({len(cluster_data)} candidatos):")
        print(f"   Perfiles: {cluster_data['perfil_oculto'].value_counts().to_dict()}")
        print(f"   Exp promedio: {cluster_data['años_experiencia'].mean():.1f} años")
        print(f"   Score técnico: {cluster_data['test_tecnico'].mean():.1f}")
        print(f"   Liderazgo: {cluster_data['liderazgo'].mean():.1f}")
    
    return X_umap, df_analisis

# Aplicar UMAP a selección de personal
X_umap_personal, df_umap_analisis = umap_seleccion_personal(df_candidatos)

