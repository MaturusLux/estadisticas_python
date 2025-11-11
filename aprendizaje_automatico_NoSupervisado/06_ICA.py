import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import seaborn as sns

plt.style.use('seaborn-v0_8')

# 1. Descarga de datos financieros

# Definir tickers de diferentes sectores y mercados
tickers = [
    'AAPL', 'MSFT', 'GOOGL',  # Tech USA
    'JPM', 'BAC', 'GS',       # Financieras USA  
    'XOM', 'CVX',             # Energía
    'WMT', 'AMZN',            # Consumo
    'TSLA', 'F',              # Automotriz
    '^GSPC', '^IXIC',         # Índices mercado
    'EURUSD=X', 'GBPUSD=X',   # Forex
    'GC=F', 'CL=F'            # Commodities
]

# Descargar datos históricos
def descargar_datos(tickers, periodo='1y'):
    datos = {}
    for ticker in tickers:
        try:
            temp = yf.download(ticker, period=periodo, interval='1d')
            datos[ticker] = temp['Close']
            print(f"Descargado: {ticker}")
        except:
            print(f"Error con: {ticker}")
    
    return pd.DataFrame(datos)

# Descargar datos
df_precios = descargar_datos(tickers, periodo='2y')
df_precios = df_precios.dropna()

# Calcular retornos diarios
df_retornos = df_precios.pct_change().dropna()
print(f"Dimensiones de retornos: {df_retornos.shape}")

# 2. Aplicar ICA para separar factores

# Estandarizar los datos
scaler = StandardScaler()
retornos_estandarizados = scaler.fit_transform(df_retornos)

# Aplicar ICA para extraer 5 factores independientes
n_componentes = 5
ica = FastICA(n_components=n_componentes, random_state=42, max_iter=1000)
componentes_ica = ica.fit_transform(retornos_retornos_estandarizados)

# Crear DataFrame con los componentes
componentes_df = pd.DataFrame(
    componentes_ica,
    index=df_retornos.index,
    columns=[f'Factor_ICA_{i+1}' for i in range(n_componentes)]
)

# 3. Interpretación de los factores ICA

# Matriz de mezcla (cómo se combinan los factores en cada activo)
matriz_mezcla = ica.mixing_

# Crear heatmap de contribuciones
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
contribuciones = pd.DataFrame(
    matriz_mezcla,
    index=df_retornos.columns,
    columns=[f'Factor_{i+1}' for i in range(n_componentes)]
)
sns.heatmap(contribuciones, cmap='RdBu_r', center=0, annot=True, fmt='.3f')
plt.title('Contribución de Factores ICA por Activo')
plt.xticks(rotation=45)

# Series temporales de los factores
plt.subplot(2, 2, 2)
for i in range(n_componentes):
    plt.plot(componentes_df.index, componentes_df[f'Factor_ICA_{i+1}'], 
             label=f'Factor {i+1}', alpha=0.8)
plt.title('Factores ICA Extraídos')
plt.legend()
plt.xticks(rotation=45)

# Distribución de los factores
plt.subplot(2, 2, 3)
for i in range(n_componentes):
    sns.kdeplot(componentes_df[f'Factor_ICA_{i+1}'], label=f'Factor {i+1}')
plt.title('Distribución de Factores ICA')
plt.legend()

plt.tight_layout()
plt.show()

# 4. Análisis de correlación e interpretación económica

# Correlación entre factores ICA (deberían ser ~0)
correlacion_factores = componentes_df.corr()
print("Matriz de correlación entre factores ICA:")
print(correlacion_factores.round(4))

# Interpretar factores con activos de referencia
def interpretar_factores(componentes_df, df_retornos):
    """Interpretar factores ICA basado en correlaciones con activos conocidos"""
    
    factores_interpretados = {}
    
    for i in range(n_componentes):
        factor = componentes_df.iloc[:, i]
        
        # Calcular correlaciones con activos clave
        correlaciones = {}
        for activo in ['AAPL', 'JPM', 'XOM', '^GSPC', 'GC=F', 'EURUSD=X']:
            if activo in df_retornos.columns:
                correlaciones[activo] = np.corrcoef(factor, df_retornos[activo])[0, 1]
        
        # Interpretar basado en correlaciones más altas
        correlaciones_abs = {k: abs(v) for k, v in correlaciones.items()}
        activo_max_corr = max(correlaciones_abs, key=correlaciones_abs.get)
        
        if 'AAPL' in activo_max_corr or 'MSFT' in activo_max_corr or 'GOOGL' in activo_max_corr:
            interpretacion = "Factor Tecnológico"
        elif 'JPM' in activo_max_corr or 'BAC' in activo_max_corr:
            interpretacion = "Factor Financiero"
        elif 'XOM' in activo_max_corr or 'CVX' in activo_max_corr:
            interpretacion = "Factor Energía"
        elif '^GSPC' in activo_max_corr:
            interpretacion = "Factor Mercado General"
        elif 'GC=F' in activo_max_corr:
            interpretacion = "Factor Oro/Commodities"
        elif 'EURUSD' in activo_max_corr:
            interpretacion = "Factor Divisas"
        else:
            interpretacion = "Factor Mixto"
            
        factores_interpretados[f'Factor_ICA_{i+1}'] = {
            'interpretacion': interpretacion,
            'correlaciones': correlaciones,
            'volatilidad': factor.std()
        }
    
    return factores_interpretados

# Obtener interpretación
interpretaciones = interpretar_factores(componentes_df, df_retornos)

print("\n INTERPRETACIÓN DE FACTORES ICA:")
print("="*50)
for factor, info in interpretaciones.items():
    print(f"\n{factor}: {info['interpretacion']}")
    print(f"Volatilidad: {info['volatilidad']:.4f}")
    print("Correlaciones principales:")
    for activo, corr in list(info['correlaciones'].items())[:3]:
        print(f"  {activo}: {corr:.3f}")

# 5. Reconstrucción y aplicación práctica

# Reconstruir retornos usando solo los factores principales
def reconstruir_retornos(ica, componentes, n_factores=None):
    """Reconstruir retornos usando subconjunto de factores"""
    if n_factores is None:
        n_factores = componentes.shape[1]
    
    # Usar solo los primeros n_factores
    componentes_reducidos = componentes[:, :n_factores]
    matriz_mezcla_reducida = ica.mixing_[:, :n_factores]
    
    # Reconstruir
    reconstruccion = np.dot(componentes_reducidos, matriz_mezcla_reducida.T)
    return reconstruccion

# Comparar reconstrucción con diferentes números de factores
plt.figure(figsize=(15, 5))

# Retorno original de AAPL
retorno_aapl = df_retornos['AAPL'].values

for i, n_factores in enumerate([2, 3, 5], 1):
    plt.subplot(1, 3, i)
    
    # Reconstruir con n_factores
    reconstruccion = reconstruir_retornos(ica, componentes_ica, n_factores)
    retorno_reconstruido = reconstruccion[:, list(df_retornos.columns).index('AAPL')]
    
    plt.plot(retorno_aapl[:100], label='Original', alpha=0.7)
    plt.plot(retorno_reconstruido[:100], label=f'Reconstruido ({n_factores} factores)', alpha=0.8)
    plt.title(f'Reconstrucción AAPL con {n_factores} factores ICA')
    plt.legend()

plt.tight_layout()
plt.show()

# 6. Aplicación: Detección de anomalías

# Usar ICA para detectar eventos anómalos en mercados
def detectar_anomalias_ica(componentes_df, umbral=3):
    """Detectar periodos donde algún factor se desvía significativamente"""
    
    anomalias = {}
    
    for factor in componentes_df.columns:
        z_scores = (componentes_df[factor] - componentes_df[factor].mean()) / componentes_df[factor].std()
        eventos_anomalos = z_scores[abs(z_scores) > umbral]
        
        anomalias[factor] = {
            'eventos': eventos_anomalos,
            'count': len(eventos_anomalos)
        }
    
    return anomalias

# Ejecutar detección de anomalías
anomalias = detectar_anomalias_ica(componentes_df)

print("\n DETECCIÓN DE ANOMALÍAS POR FACTOR ICA:")
print("="*50)
for factor, info in anomalias.items():
    print(f"\n{factor}: {info['count']} eventos anómalos")
    if not info['eventos'].empty:
        print(f"  Fechas clave: {info['eventos'].index.strftime('%Y-%m-%d').tolist()[:3]}")
#####################################################
#  Interpretación Económica de los Factores ICA
# Típicamente encontrarás factores como:

# Factor Mercado General: Correlacionado con índices como S&P500
# Factor Sectorial: Tecnológico, Financiero, Energético
# Factor Divisas: Relacionado con movimientos cambiarios
# Factor Commodities: Oro, petróleo, materias primas
# Factor Específico: Ruido o factores idiosincráticos

#  Ventajas de ICA en Finanzas

# Separación real de fuentes: Encuentra factores estadísticamente independientes
# Mejor que PCA: PCA solo garantiza no correlación, ICA independencia
# Robusto a no normalidad: Funciona bien con distribuciones financieras (colas pesadas)
# Interpretabilidad: Los factores suelen corresponder a drivers económicos reales
# Este enfoque es particularmente útil para gestión de riesgo, construcción de portafolios y detección de regímenes de mercado.
