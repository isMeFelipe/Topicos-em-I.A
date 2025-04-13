import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from itertools import combinations

# Carregar os dados
# Dados da quality = 3 a 8
df = pd.read_csv("dataset.csv", sep=";")
print("Colunas disponíveis:", df.columns.tolist())

# Definir combinações de features para testar
feature_combinations = [
    # Combinações básicas
    ['fixed acidity', 'volatile acidity', 'alcohol'],
    ['fixed acidity', 'pH', 'alcohol'],
    ['citric acid', 'residual sugar', 'alcohol'],
    ['density', 'alcohol', 'chlorides'],
    
    # Combinações relacionadas à acidez
    ['fixed acidity', 'volatile acidity', 'citric acid', 'pH', 'alcohol'],
    ['fixed acidity', 'citric acid', 'pH'],
    ['volatile acidity', 'citric acid', 'sulphates'],
    
    # Combinações relacionadas à doçura e corpo
    ['residual sugar', 'alcohol', 'density'],
    ['residual sugar', 'alcohol', 'chlorides'],
    ['residual sugar', 'free sulfur dioxide', 'total sulfur dioxide'],
    
    # Combinações relacionadas a compostos sulfurados
    ['free sulfur dioxide', 'total sulfur dioxide', 'sulphates'],
    ['chlorides', 'sulphates', 'alcohol'],
    ['free sulfur dioxide', 'total sulfur dioxide', 'pH'],
    
    # Combinações mais complexas
    ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'alcohol', 'sulphates'],
    ['fixed acidity', 'pH', 'alcohol', 'density', 'sulphates'],
    ['volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'total sulfur dioxide'],
    
    # Combinações específicas para características distintas
    ['alcohol', 'density', 'pH', 'sulphates'],  # Corpo e estrutura
    ['fixed acidity', 'volatile acidity', 'alcohol', 'sulphates'],  # Acidez e preservação
    ['residual sugar', 'alcohol', 'chlorides', 'total sulfur dioxide'],  # Doçura e conservantes
    
    # Todas as features (exceto quality)
    [col for col in df.columns if col != 'quality']
]

# Adicionar todas as colunas (exceto quality) como uma combinação
all_features = [col for col in df.columns if col != 'quality']
feature_combinations.append(all_features)

# Dicionário para armazenar os melhores resultados de cada combinação
results = {}

for i, features in enumerate(feature_combinations, 1):
    print(f"\n🔍 Testando combinação {i}: {features}")
    
    # Preparar dados
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Redução de dimensionalidade com PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Testar K-Means
    print("\nK-Means ---------------------")
    best_kmeans = {'silhouette': -1, 'k': None}
    
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        silhouette = silhouette_score(X_scaled, labels)
        
        if silhouette > best_kmeans['silhouette']:
            best_kmeans['silhouette'] = silhouette
            best_kmeans['k'] = k
            best_kmeans['labels'] = labels
            
        print(f'Silhouette para K={k}: {silhouette:.4f}')
    
    print(f"Melhor K para K-Means: {best_kmeans['k']}, Silhouette: {best_kmeans['silhouette']:.3f}")
    
    # Testar DBSCAN
    print("\nDBSCAN ---------------------")
    best_dbscan = {'silhouette': -1, 'eps': None, 'min_samples': None}
    
    for eps in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        for min_samples in [3, 5, 7, 10]:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            if n_clusters > 1:
                valid_points = labels != -1
                if sum(valid_points) > 0:
                    silhouette = silhouette_score(X_scaled[valid_points], labels[valid_points])
                    
                    if silhouette > best_dbscan['silhouette']:
                        best_dbscan['silhouette'] = silhouette
                        best_dbscan['eps'] = eps
                        best_dbscan['min_samples'] = min_samples
                        best_dbscan['labels'] = labels
                        best_dbscan['n_clusters'] = n_clusters
                    
                    print(f'eps={eps}, min_samples={min_samples}: clusters={n_clusters}, silhouette={silhouette:.4f}')
    
    print(f"Melhores params para DBSCAN: eps={best_dbscan['eps']}, min_samples={best_dbscan['min_samples']}")
    print(f"Clusters encontrados: {best_dbscan['n_clusters']}, Silhouette: {best_dbscan['silhouette']:.3f}")
    
    # Determinar qual método foi melhor
    if best_kmeans['silhouette'] > best_dbscan['silhouette']:
        best_method = 'K-Means'
        best_score = best_kmeans['silhouette']
    else:
        best_method = 'DBSCAN'
        best_score = best_dbscan['silhouette']
    
    print(f"\n🏆 Melhor método para combinação {i}: {best_method} (Silhouette: {best_score:.3f})")
    
    # Armazenar resultados
    results[f"Combinação {i}"] = {
        'features': features,
        'best_method': best_method,
        'best_score': best_score,
        'kmeans': best_kmeans,
        'dbscan': best_dbscan
    }
    
    # Visualizar resultados
    plt.figure(figsize=(15, 6))
    
    # Plot K-Means
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=best_kmeans['labels'], palette='tab10')
    plt.title(f'K-Means (k={best_kmeans["k"]}, Silhouette={best_kmeans["silhouette"]:.3f})')
    
    # Plot DBSCAN
    plt.subplot(1, 2, 2)
    if best_dbscan['labels'] is not None:
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=best_dbscan['labels'], palette='tab10')
    plt.title(f'DBSCAN (eps={best_dbscan["eps"]}, min_samples={best_dbscan["min_samples"]}\nSilhouette={best_dbscan["silhouette"]:.3f}, Clusters={best_dbscan["n_clusters"]}')
    
    plt.suptitle(f"Combinação de Features: {', '.join(features)}")
    plt.tight_layout()
    plt.show()

# Mostrar resumo de todos os testes
print("\n📊 RESUMO DE TODOS OS TESTES:")
for name, result in results.items():
    print(f"\n{name}: {', '.join(result['features'])}")
    print(f"Melhor método: {result['best_method']} (Score: {result['best_score']:.3f})")
    print(f"K-Means - k: {result['kmeans']['k']}, Silhouette: {result['kmeans']['silhouette']:.3f}")
    if result['dbscan']['eps'] is not None:
        print(f"DBSCAN - eps: {result['dbscan']['eps']}, min_samples: {result['dbscan']['min_samples']}, Clusters: {result['dbscan']['n_clusters']}, Silhouette: {result['dbscan']['silhouette']:.3f}")
    else:
        print("DBSCAN - Nenhum agrupamento válido encontrado")