#  Ejemplo t-SNE en Selección de Personal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Configuración
plt.style.use('seaborn-v0_8')
np.random.seed(42)

# 1. Simulación de datos de candidatos
def simular_candidatos_agronomos(n_candidatos=200):
    """Simular datos complejos de candidatos para Ingeniero Agrónomo"""
    
    # Características de alta dimensión (30 features)
    np.random.seed(42)
    
    datos = {}
    
    # 1. Experiencia y formación (5 features)
    datos['años_experiencia'] = np.random.normal(10, 3, n_candidatos)
    datos['grado_academico'] = np.random.choice([1, 2, 3], n_candidatos, p=[0.3, 0.5, 0.2])  # 1: Lic, 2: MSc, 3: PhD
    datos['certificaciones'] = np.random.poisson(3, n_candidatos)
    datos['cursos_especializacion'] = np.random.poisson(5, n_candidatos)
    datos['idiomas'] = np.random.choice([1, 2, 3, 4], n_candidatos, p=[0.2, 0.4, 0.3, 0.1])
    
    # 2. Habilidades técnicas específicas (10 features)
    habilidades_tecnicas = [
        'manejo_suelos', 'riego_eficiente', 'fertirriego', 'biotecnologia',
        'agricultura_precision', 'manejo_plagas', 'cultivos_extensivos',
        'ganaderia', 'agroindustria', 'sustentabilidad'
    ]
    
    for habilidad in habilidades_tecnicas:
        # Crear correlaciones locales (candidatos similares en grupos específicos)
        base = np.random.normal(7, 2, n_candidatos)
        # Añadir patrones locales
        grupo = np.random.choice([0, 1, 2, 3], n_candidatos, p=[0.25, 0.25, 0.25, 0.25])
        patron = np.where(groupe == 0, 1.5, 
                 np.where(grupo == 1, -1.0, 
                 np.where(grupo == 2, 0.5, -0.8)))
        datos[habilidad] = np.clip(base + patron, 1, 10)
    
    # 3. Habilidades blandas (8 features)
    habilidades_blandas = [
        'liderazgo', 'comunicacion', 'negociacion', 'resolucion_problemas',
        'trabajo_equipo', 'adaptabilidad', 'innovacion', 'gestion_proyectos'
    ]
    
    for habilidad in habilidades_blandas:
        datos[habilidad] = np.random.normal(7, 1.5, n_candidatos)
    
    # 4. Especializaciones geográficas (3 features)
    datos['experiencia_pampa'] = np.random.normal(8, 1.5, n_candidatos)
    datos['experiencia_noroeste'] = np.random.normal(5, 2, n_candidatos)
    datos['experiencia_patagonia'] = np.random.normal(4, 2, n_candidatos)
    
    # 5. Resultados de evaluación (4 features)
    datos['test_tecnico'] = np.random.normal(75, 15, n_candidatos)
    datos['test_psicometrico'] = np.random.normal(70, 12, n_candidatos)
    datos['entrevista_tecnica'] = np.random.normal(80, 10, n_candidatos)
    datos['caso_practico'] = np.random.normal(75, 12, n_candidatos)
    
    df = pd.DataFrame(datos)
    
    # Crear grupos naturales (no lineales) - ESTRUCTURA LOCAL
    # Estos grupos existen pero no son linealmente separables
    condiciones_complejas = (
        (df['manejo_suelos'] > 7) & (df['agricultura_precision'] > 8) & (df['experiencia_pampa'] > 7) |
        (df['biotecnologia'] > 8) & (df['innovacion'] > 8) & (df['grado_academico'] == 3) |
        (df['ganaderia'] > 7) & (df['experiencia_patagonia'] > 6) & (df['adaptabilidad'] > 7)
    )
    
    df['perfil_oculto'] = np.where(condiciones_complejas, 'Especialista_Avanzado',
                          np.where(df['test_tecnico'] > 85, 'Tecnico_Destacado',
                          np.where(df['liderazgo'] > 8, 'Perfil_Gerencial', 'Generalista')))
    
    return df

# Generar datos
df_candidatos = simular_candidatos_agronomos(200)
print("📊 DATOS DE CANDIDATOS - ESTRUCTURA COMPLEJA")
print(f"Dimensiones: {df_candidatos.shape}")
print(f"Variables: {list(df_candidatos.columns)}")
print(f"\nPerfiles ocultos encontrados:")
print(df_candidatos['perfil_oculto'].value_counts())

# 2. Aplicación de t-SNE para descubrir estructura local
def analisis_tsne_seleccion_personal(df):
    """Aplicar t-SNE para descubrir grupos naturales en candidatos"""
    
    # Preparar datos (excluir la variable objetivo)
    X = df.drop('perfil_oculto', axis=1)
    y = df['perfil_oculto']
    
    # Estandarizar
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    print("🎯 APLICANDO t-SNE PARA DESCUBRIR ESTRUCTURA LOCAL")
    print("="*60)
    
    # Aplicar t-SNE con diferentes perplexities
    perplexities = [5, 15, 30, 50]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    resultados_tsne = {}
    
    for i, perplexity in enumerate(perplexities):
        tsne = TSNE(n_components=2, random_state=42, 
                   perplexity=perplexity, n_iter=1000)
        X_tsne = tsne.fit_transform(X_std)
        
        resultados_tsne[perplexity] = X_tsne
        
        # Visualizar
        scatter = axes[i].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                c=pd.factorize(y)[0], cmap='viridis', 
                                alpha=0.7, s=60)
        axes[i].set_title(f't-SNE - Perplexity: {perplexity}\nEstructura Local Revelada')
        axes[i].set_xlabel('t-SNE 1')
        axes[i].set_ylabel('t-SNE 2')
        
        # Calcular métrica de calidad
        sil_score = silhouette_score(X_tsne, pd.factorize(y)[0])
        axes[i].text(0.02, 0.98, f'Silhouette: {sil_score:.3f}', 
                    transform=axes[i].transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return resultados_tsne, X_std, y

# Ejecutar análisis t-SNE
resultados_tsne, X_std, y = analisis_tsne_seleccion_personal(df_candidatos)

# 3. Comparación t-SNE vs PCA
def comparar_tsne_pca(X_std, y):
    """Comparar cómo t-SNE y PCA revelan diferente información"""
    
    from sklearn.decomposition import PCA
    
    # Aplicar PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)
    
    # Aplicar t-SNE óptimo
    tsne_optimo = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne_optimo.fit_transform(X_std)
    
    # Visualizar comparación
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # PCA
    scatter_pca = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], 
                                c=pd.factorize(y)[0], cmap='viridis', 
                                alpha=0.7, s=50)
    axes[0].set_title('PCA - Vista Global\n(Varianza Explicada)')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    
    # t-SNE
    scatter_tsne = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                 c=pd.factorize(y)[0], cmap='viridis', 
                                 alpha=0.7, s=50)
    axes[1].set_title('t-SNE - Vista Local\n(Estructura de Vecindarios)')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    
    # Combinación para insights
    for i, perfil in enumerate(y.unique()):
        mask = y == perfil
        axes[2].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       label=f'{perfil} (PCA)', alpha=0.5, s=40,
                       marker='o')
        axes[2].scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       label=f'{perfil} (t-SNE)', alpha=0.5, s=40,
                       marker='s')
    
    axes[2].set_title('Comparación PCA vs t-SNE\n(Patrones de Grupos)')
    axes[2].set_xlabel('Componente 1')
    axes[2].set_ylabel('Componente 2')
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Métricas comparativas
    sil_pca = silhouette_score(X_pca, pd.factorize(y)[0])
    sil_tsne = silhouette_score(X_tsne, pd.factorize(y)[0])
    
    print("📊 COMPARACIÓN CUANTITATIVA:")
    print("="*50)
    print(f"PCA - Silhouette Score:  {sil_pca:.4f}")
    print(f"t-SNE - Silhouette Score: {sil_tsne:.4f}")
    print(f"Mejora t-SNE: {((sil_tsne - sil_pca) / sil_pca * 100):.1f}%")
    
    return X_pca, X_tsne

# Ejecutar comparación
X_pca, X_tsne = comparar_tsne_pca(X_std, y)

# 4. Interpretación de resultados para RRHH
def interpretacion_rh_tsne(df, X_tsne, y):
    """Interpretar los resultados de t-SNE para decisiones de RRHH"""
    
    # Crear DataFrame con resultados
    df_resultados = df.copy()
    df_resultados['tsne_1'] = X_tsne[:, 0]
    df_resultados['tsne_2'] = X_tsne[:, 1]
    
    # Análisis por perfil
    print("🎯 INTERPRETACIÓN PARA SELECCIÓN DE PERSONAL")
    print("="*60)
    
    perfiles = y.unique()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Mapa general de talento
    for i, perfil in enumerate(perfiles):
        mask = df_resultados['perfil_oculto'] == perfil
        axes[0,0].scatter(df_resultados.loc[mask, 'tsne_1'], 
                         df_resultados.loc[mask, 'tsne_2'], 
                         label=perfil, alpha=0.8, s=60)
    
    axes[0,0].set_title('Mapa de Talento - t-SNE\n(Grupos Naturales de Candidatos)')
    axes[0,0].set_xlabel('t-SNE 1')
    axes[0,0].set_ylabel('t-SNE 2')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Características por grupo
    caracteristicas_clave = ['años_experiencia', 'test_tecnico', 'liderazgo', 'innovacion']
    
    for i, caracteristica in enumerate(caracteristicas_clave):
        scatter = axes[0,1].scatter(df_resultados['tsne_1'], df_resultados['tsne_2'], 
                                  c=df_resultados[caracteristica], cmap='plasma', 
                                  alpha=0.7, s=50)
        axes[0,1].set_title(f'Mapa por {caracteristica.replace("_", " ").title()}')
        axes[0,1].set_xlabel('t-SNE 1')
        axes[0,1].set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=axes[0,1], label=caracteristica)
        break  # Solo mostrar una para ejemplo
    
    # 3. Análisis de clusters para recomendaciones
    from sklearn.cluster import DBSCAN
    
    # Usar DBSCAN para encontrar grupos densos
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(X_tsne)
    df_resultados['cluster_tsne'] = clustering.labels_
    
    # 4. Recomendaciones específicas por cluster
    axes[1,0].scatter(df_resultados['tsne_1'], df_resultados['tsne_2'], 
                     c=df_resultados['cluster_tsne'], cmap='tab10', 
                     alpha=0.7, s=50)
    axes[1,0].set_title('Clusters de Similitud - t-SNE\n(Grupos para Estrategias Específicas)')
    axes[1,0].set_xlabel('t-SNE 1')
    axes[1,0].set_ylabel('t-SNE 2')
    
    # 5. Identificar candidatos únicos/raros
    distancias = np.sqrt(np.sum(X_tsne**2, axis=1))
    percentil_95 = np.percentile(distancias, 95)
    candidatos_unicos = distancias > percentil_95
    
    axes[1,1].scatter(df_resultados['tsne_1'], df_resultados['tsne_2'], 
                     c='lightgray', alpha=0.5, s=40, label='Candidatos típicos')
    axes[1,1].scatter(df_resultados.loc[candidatos_unicos, 'tsne_1'], 
                     df_resultados.loc[candidatos_unicos, 'tsne_2'], 
                     c='red', s=80, label='Perfiles únicos', marker='*')
    axes[1,1].set_title('Identificación de Perfiles Únicos\n(Talento Diferenciador)')
    axes[1,1].set_xlabel('t-SNE 1')
    axes[1,1].set_ylabel('t-SNE 2')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Análisis estratégico para RRHH
    print("\n💡 RECOMENDACIONES ESTRATÉGICAS BASADAS EN t-SNE:")
    print("="*60)
    
    # Por perfil oculto
    for perfil in perfiles:
        mask = df_resultados['perfil_oculto'] == perfil
        n_candidatos = mask.sum()
        avg_experiencia = df_resultados.loc[mask, 'años_experiencia'].mean()
        avg_tecnico = df_resultados.loc[mask, 'test_tecnico'].mean()
        
        print(f"\n🔹 {perfil}:")
        print(f"   • {n_candidatos} candidatos ({n_candidatos/len(df_resultados)*100:.1f}%)")
        print(f"   • Experiencia promedio: {avg_experiencia:.1f} años")
        print(f"   • Puntaje técnico: {avg_tecnico:.1f}/100")
        
        # Recomendación específica
        if perfil == 'Especialista_Avanzado':
            print("   → Estrategia: Retener y desarrollar - alto valor")
        elif perfil == 'Tecnico_Destacado':
            print("   → Estrategia: Asignar roles técnicos críticos")
        elif perfil == 'Perfil_Gerencial':
            print("   → Estrategia: Preparar para liderazgo")
        else:
            print("   → Estrategia: Desarrollo y rotación")
    
    # Candidatos únicos
    print(f"\n🌟 PERFILES ÚNICOS IDENTIFICADOS: {candidatos_unicos.sum()}")
    print("Estos candidatos tienen combinaciones de habilidades poco comunes")
    
    return df_resultados

# Ejecutar interpretación
df_analisis_final = interpretacion_rh_tsne(df_candidatos, X_tsne, y)


