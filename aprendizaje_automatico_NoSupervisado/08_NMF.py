import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.datasets import make_blobs, load_iris, fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Configuración
plt.style.use('seaborn-v0_8')
np.random.seed(42)

#############
def ejemplo_basico_nmf():
    """Ejemplo básico de factorización NMF"""
    
    # Crear matriz de datos no negativos
    V = np.array([
        [1, 2, 3, 0, 1],
        [4, 5, 6, 1, 2],
        [7, 8, 9, 2, 3],
        [0, 1, 2, 3, 4],
        [1, 0, 1, 4, 5]
    ], dtype=float)
    
    print("📊 MATRIZ ORIGINAL V (5x5):")
    print(V)
    print(f"Rango de valores: [{V.min():.1f}, {V.max():.1f}]")
    
    # Aplicar NMF
    n_components = 2
    model = NMF(n_components=n_components, init='random', random_state=42)
    W = model.fit_transform(V)  # Matriz de bases
    H = model.components_       # Matriz de coeficientes
    
    # Reconstruir matriz
    V_reconstructed = np.dot(W, H)
    
    print(f"\n🎯 FACTORIZACIÓN NMF CON {n_components} COMPONENTES")
    print("\nMatriz W (Bases - 5x2):")
    print(W.round(3))
    print("\nMatriz H (Coeficientes - 2x5):")
    print(H.round(3))
    
    print("\n🔁 RECONSTRUCCIÓN V ≈ W * H:")
    print(V_reconstructed.round(3))
    
    print(f"\n📈 ERROR DE RECONSTRUCCIÓN: {model.reconstruction_err_:.4f}")
    
    # Visualizar
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Matriz original
    im1 = axes[0].imshow(V, cmap='YlOrRd', aspect='auto')
    axes[0].set_title('Matriz Original V')
    plt.colorbar(im1, ax=axes[0])
    
    # Matriz reconstruida
    im2 = axes[1].imshow(V_reconstructed, cmap='YlOrRd', aspect='auto')
    axes[1].set_title('Matriz Reconstruida W*H')
    plt.colorbar(im2, ax=axes[1])
    
    # Error
    error = np.abs(V - V_reconstructed)
    im3 = axes[2].imshow(error, cmap='Reds', aspect='auto')
    axes[2].set_title('Error |V - W*H|')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()
    
    return V, W, H, V_reconstructed

# Ejecutar ejemplo básico
V, W, H, V_recon = ejemplo_basico_nmf()

############

def ejemplo_analisis_productos():
    """Ejemplo de NMF para análisis de productos y características"""
    
    # Simular datos de productos y características
    productos = ['Laptop Gamer', 'Tablet Profesional', 'Smartphone Flagship', 
                'Laptop Oficina', 'Smartphone Económico', 'Tablet Básica']
    
    caracteristicas = ['Procesamiento', 'Memoria RAM', 'Almacenamiento', 
                      'Batería', 'Pantalla', 'Precio', 'Cámara', 'Portabilidad']
    
    # Matriz: productos x características (valores 0-10)
    np.random.seed(42)
    datos_productos = np.array([
        [9, 8, 7, 4, 8, 2, 6, 3],  # Laptop Gamer
        [7, 6, 6, 8, 7, 5, 7, 9],  # Tablet Profesional
        [8, 7, 6, 6, 8, 3, 9, 8],  # Smartphone Flagship
        [6, 5, 5, 6, 6, 7, 4, 4],  # Laptop Oficina
        [4, 3, 4, 7, 5, 9, 5, 8],  # Smartphone Económico
        [3, 2, 3, 6, 4, 8, 3, 9]   # Tablet Básica
    ])
    
    df_productos = pd.DataFrame(datos_productos, 
                               index=productos, 
                               columns=caracteristicas)
    
    print("📊 MATRIZ PRODUCTOS x CARACTERÍSTICAS:")
    print(df_productos)
    
    # Aplicar NMF
    n_componentes = 3
    model = NMF(n_components=n_componentes, init='nndsvd', random_state=42)
    W = model.fit_transform(datos_productos)  # Productos x Componentes
    H = model.components_                     # Componentes x Características
    
    # Crear DataFrames para interpretación
    df_W = pd.DataFrame(W, index=productos, 
                       columns=[f'Patron_{i+1}' for i in range(n_componentes)])
    
    df_H = pd.DataFrame(H, index=[f'Patron_{i+1}' for i in range(n_componentes)], 
                       columns=caracteristicas)
    
    print(f"\n🎯 FACTORIZACIÓN NMF - {n_componentes} PATRONES LATENTES")
    print("\nMatriz W (Productos x Patrones):")
    print(df_W.round(3))
    
    print("\nMatriz H (Patrones x Características):")
    print(df_H.round(3))
    
    # Interpretar patrones
    print("\n🔍 INTERPRETACIÓN DE PATRONES:")
    for i in range(n_componentes):
        print(f"\n📊 PATRÓN {i+1}:")
        # Características más importantes del patrón
        caracteristicas_importantes = df_H.iloc[i].nlargest(3)
        for car, valor in caracteristicas_importantes.items():
            print(f"   • {car}: {valor:.3f}")
        
        # Productos que mejor representan este patrón
        productos_representativos = df_W.iloc[:, i].nlargest(2)
        print(f"   Productos representativos: {', '.join(productos_representativos.index)}")
    
    # Visualizar
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Heatmap matriz original
    sns.heatmap(df_productos, annot=True, cmap='YlOrRd', ax=axes[0,0])
    axes[0,0].set_title('Matriz Original\nProductos x Características')
    
    # Heatmap matriz W
    sns.heatmap(df_W, annot=True, cmap='Blues', ax=axes[0,1])
    axes[0,1].set_title('Matriz W\nProductos x Patrones Latentes')
    
    # Heatmap matriz H
    sns.heatmap(df_H, annot=True, cmap='Greens', ax=axes[1,0])
    axes[1,0].set_title('Matriz H\nPatrones Latentes x Características')
    
    # Productos en espacio de componentes
    for i, producto in enumerate(productos):
        axes[1,1].scatter(W[i, 0], W[i, 1], s=100, alpha=0.7)
        axes[1,1].annotate(producto, (W[i, 0], W[i, 1]), xytext=(5, 5), 
                          textcoords='offset points', fontsize=9)
    
    axes[1,1].set_xlabel('Patrón Latente 1')
    axes[1,1].set_ylabel('Patrón Latente 2')
    axes[1,1].set_title('Productos en Espacio de Patrones Latentes')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return df_productos, df_W, df_H

# Ejecutar análisis de productos
df_prod, df_W, df_H = ejemplo_analisis_productos()

################
def ejemplo_analisis_texto():
    """Ejemplo de NMF para análisis de texto y topic modeling"""
    
    # Documentos de ejemplo (noticias sobre tecnología)
    documentos = [
        "apple iphone smartphone tecnologia innovacion camara pantalla",
        "samsung android telefono movil aplicaciones google play store",
        "microsoft windows software computadora office excel word",
        "nvidia gpu tarjeta grafica juegos rendimiento computadora",
        "intel procesador cpu computadora rendimiento velocidad",
        "amd ryzen procesador gaming computadora rendimiento",
        "google android smartphone aplicaciones inteligencia artificial",
        "apple macbook computadora portatil diseño rendimiento",
        "samsung tablet pantalla aplicaciones android movil",
        "microsoft surface tablet windows software oficina"
    ]
    
    categorias_reales = ['apple', 'samsung', 'microsoft', 'nvidia', 'intel', 
                        'amd', 'google', 'apple', 'samsung', 'microsoft']
    
    # Crear matriz término-documento con TF-IDF
    vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
    X = vectorizer.fit_transform(documentos).toarray()
    
    términos = vectorizer.get_feature_names_out()
    
    print("📝 DOCUMENTOS DE EJEMPLO:")
    for i, doc in enumerate(documentos):
        print(f"Doc {i+1}: {doc}")
    
    print(f"\n📊 MATRIZ TF-IDF ({X.shape[0]} documentos x {X.shape[1]} términos)")
    
    # Aplicar NMF para encontrar topics
    n_topics = 4
    model = NMF(n_components=n_topics, init='nndsvd', random_state=42)
    W = model.fit_transform(X)  # Documentos x Topics
    H = model.components_       # Topics x Términos
    
    # Crear DataFrames
    df_W = pd.DataFrame(W, index=[f'Doc_{i+1}' for i in range(len(documentos))],
                       columns=[f'Topic_{i+1}' for i in range(n_topics)])
    
    df_H = pd.DataFrame(H, index=[f'Topic_{i+1}' for i in range(n_topics)],
                       columns=términos)
    
    print(f"\n🎯 NMF TOPIC MODELING - {n_topics} TEMAS")
    
    # Mostrar palabras clave por topic
    print("\n🔤 PALABRAS CLAVE POR TEMA:")
    n_palabras = 5
    for i in range(n_topics):
        palabras_topic = df_H.iloc[i].nlargest(n_palabras)
        print(f"\n📚 Tema {i+1}:")
        palabras_str = ", ".join([f"{palabra}({peso:.3f})" 
                                for palabra, peso in palabras_topic.items()])
        print(f"   {palabras_str}")
    
    # Mostrar documentos y sus temas principales
    print("\n📄 DOCUMENTOS Y TEMAS PRINCIPALES:")
    for i, doc in enumerate(documentos):
        topic_principal = np.argmax(W[i])
        peso_principal = W[i, topic_principal]
        categoria_real = categorias_reales[i]
        
        print(f"Doc {i+1}: Tema {topic_principal+1} (peso: {peso_principal:.3f}) - Categoría real: {categoria_real}")
    
    # Visualizar
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Distribución de temas en documentos
    df_W.plot(kind='bar', ax=axes[0,0], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[0,0].set_title('Distribución de Temas en Documentos')
    axes[0,0].set_ylabel('Peso del Tema')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Heatmap de términos por tema
    sns.heatmap(df_H.T, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[0,1])
    axes[0,1].set_title('Pesos de Términos por Tema')
    
    # Tópicos en espacio 2D (usando los dos temas principales)
    colors = ['red', 'blue', 'green', 'orange']
    for i in range(len(documentos)):
        color_idx = list(set(categorias_reales)).index(categorias_reales[i])
        axes[1,0].scatter(W[i, 0], W[i, 1], color=colors[color_idx], s=100, alpha=0.7)
        axes[1,0].annotate(f'Doc{i+1}', (W[i, 0], W[i, 1]), xytext=(5, 5), 
                          textcoords='offset points', fontsize=8)
    
    axes[1,0].set_xlabel('Tema 1')
    axes[1,0].set_ylabel('Tema 2')
    axes[1,0].set_title('Documentos en Espacio de Temas')
    axes[1,0].grid(True, alpha=0.3)
    
    # Barras de temas principales por documento
    temas_principales = np.argmax(W, axis=1)
    pd.Series(temas_principales).value_counts().sort_index().plot(
        kind='bar', color='skyblue', ax=axes[1,1]
    )
    axes[1,1].set_title('Distribución de Temas Principales')
    axes[1,1].set_xlabel('Tema')
    axes[1,1].set_ylabel('Número de Documentos')
    
    plt.tight_layout()
    plt.show()
    
    return documentos, df_W, df_H

# Ejecutar análisis de texto
docs, df_W_text, df_H_text = ejemplo_analisis_texto()


#################

def ejemplo_procesamiento_imagenes():
    """Ejemplo de NMF para procesamiento de imágenes"""
    
    try:
        # Cargar dataset de caras (imágenes pequeñas)
        faces = fetch_olivetti_faces(shuffle=True, random_state=42)
        X = faces.data
        images = faces.images
        
        print(f"📸 DATASET DE IMÁGENES: {X.shape[0]} imágenes, {X.shape[1]} píxeles")
        print(f"Dimensiones de cada imagen: {images.shape[1]}x{images.shape[2]}")
        
        # Normalizar a [0, 1] (NMF requiere no negatividad)
        X_normalized = MinMaxScaler().fit_transform(X)
        
        # Aplicar NMF para encontrar componentes faciales
        n_components = 9
        model = NMF(n_components=n_components, init='nndsvda', random_state=42)
        W = model.fit_transform(X_normalized)  # Imágenes x Componentes
        H = model.components_                  # Componentes x Píxeles
        
        print(f"\n🎯 NMF IMAGENES - {n_components} COMPONENTES FACIALES")
        
        # Visualizar componentes (partes de caras)
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        
        # Mostrar algunas imágenes originales
        axes[0,0].imshow(images[0], cmap='gray')
        axes[0,0].set_title('Imagen Original 1')
        axes[0,0].axis('off')
        
        axes[0,1].imshow(images[10], cmap='gray')
        axes[0,1].set_title('Imagen Original 2')
        axes[0,1].axis('off')
        
        axes[0,2].imshow(images[20], cmap='gray')
        axes[0,2].set_title('Imagen Original 3')
        axes[0,2].axis('off')
        
        axes[0,3].axis('off')  # Espacio vacío
        
        # Mostrar componentes NMF
        for i in range(min(8, n_components)):
            row = 1 + i // 4
            col = i % 4
            component_img = H[i].reshape(images[0].shape)
            axes[row, col].imshow(component_img, cmap='gray')
            axes[row, col].set_title(f'Componente NMF {i+1}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Reconstruir una imagen usando componentes
        imagen_idx = 5
        imagen_original = images[imagen_idx]
        imagen_reconstruida = np.dot(W[imagen_idx], H).reshape(images[0].shape)
        
        # Visualizar reconstrucción
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(imagen_original, cmap='gray')
        axes[0].set_title('Imagen Original')
        axes[0].axis('off')
        
        axes[1].imshow(imagen_reconstruida, cmap='gray')
        axes[1].set_title('Imagen Reconstruida\n(W * H)')
        axes[1].axis('off')
        
        # Mostrar contribución de componentes
        contribuciones = W[imagen_idx]
        axes[2].bar(range(len(contribuciones)), contribuciones)
        axes[2].set_title('Contribución de Componentes')
        axes[2].set_xlabel('Componente NMF')
        axes[2].set_ylabel('Peso')
        
        plt.tight_layout()
        plt.show()
        
        print(f"📊 CONTRIBUCIÓN DE COMPONENTES PARA IMAGEN {imagen_idx}:")
        for i, contrib in enumerate(contribuciones):
            print(f"  Componente {i+1}: {contrib:.4f}")
        
        return X_normalized, W, H, images
        
    except Exception as e:
        print(f"Error cargando dataset de caras: {e}")
        print("Creando datos de imagen sintéticos...")
        
        # Crear datos sintéticos
        n_imagenes = 100
        height, width = 20, 20
        X_sintetico = np.random.rand(n_imagenes, height * width)
        
        # Aplicar NMF
        n_components = 6
        model = NMF(n_components=n_components, random_state=42)
        W = model.fit_transform(X_sintetico)
        H = model.components_
        
        print(f"NMF aplicado a datos sintéticos: {n_imagenes} imágenes")
        
        return X_sintetico, W, H, None

# Ejecutar procesamiento de imágenes
X_img, W_img, H_img, images = ejemplo_procesamiento_imagenes()

#########
def comparar_parametros_nmf():
    """Comparar diferentes parámetros de NMF"""
    
    # Datos de ejemplo
    X = np.random.rand(50, 20)
    
    configuraciones = [
        {'init': 'random', 'solver': 'cd'},
        {'init': 'nndsvd', 'solver': 'cd'},
        {'init': 'nndsvda', 'solver': 'cd'},
        {'init': 'nndsvdar', 'solver': 'cd'},
        {'init': 'random', 'solver': 'mu'},
    ]
    
    resultados = []
    
    for config in configuraciones:
        try:
            model = NMF(n_components=5, random_state=42, **config)
            W = model.fit_transform(X)
            
            resultados.append({
                'configuracion': str(config),
                'error_reconstruccion': model.reconstruction_err_,
                'iteraciones': model.n_iter_,
                'convergio': model.n_iter_ < model.max_iter
            })
        except Exception as e:
            print(f"Error con configuración {config}: {e}")
    
    df_resultados = pd.DataFrame(resultados)
    print("🔧 COMPARACIÓN DE PARÁMETROS NMF:")
    print(df_resultados.round(4))

# Comparar parámetros
comparar_parametros_nmf()


##############################################

# # Configuración recomendada para diferentes casos
# configuraciones_recomendadas = {
#     'texto': {
#         'init': 'nndsvd',
#         'solver': 'cd',
#         'beta_loss': 'frobenius'
#     },
#     'imagenes': {
#         'init': 'nndsvda', 
#         'solver': 'mu',
#         'beta_loss': 'kullback-leibler'
#     },
#     'datos_generales': {
#         'init': 'random',
#         'solver': 'cd',
#         'beta_loss': 'frobenius'
#     }
# }

# # Para la mayoría de casos:
# nmf_recomendado = NMF(
#     n_components=10,      # Elegir basado en el criterio del dominio
#     init='nndsvda',       # Buena inicialización
#     random_state=42,      # Reproducibilidad
#     max_iter=1000         # Suficientes iteraciones
# )
