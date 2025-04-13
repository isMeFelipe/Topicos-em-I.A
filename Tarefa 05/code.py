import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Carregar os dados
df = pd.read_csv("dataset.csv", sep=";")
print(df.columns)

# Padronizar os dados (importante para clustering)
X = df.drop("quality", axis=1)  # Retira a coluna de qualidade (quality)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means:
# Encontrar melhor k (compara o silhouette )
print("\nK-Means ---------------------")
sil_k = []
range_k = range(2, 10)
count = 2;
for k in range_k:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    sil_k.append(silhouette_score(X_scaled, labels))
    print('Silhouette para K = ',count,': ',silhouette_score(X_scaled, labels))
    count+=1;

best_k = range_k[sil_k.index(max(sil_k))]
print(f"Melhor K para K-Means: {best_k}, Silhouette: {max(sil_k):.3f}")

# Rodar K-Means final ( com melhor silhoutte)
kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)





# DBSCAN:
# Testar combinações de eps e min_samples
sil_dbscan = -1 # valor inicial do silhoutte (-1 a 1)
best_eps, best_min = None, None
best_labels_db = None

print(" \nDBSCAN ---------------------")
count = 1;
for eps in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    for min_samples in [3, 4, 5, 6, 7]:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters > 1:
            score = silhouette_score(X_scaled, labels)
            print('score rodada ', count, ': ', score)
            count += 1
            if score > sil_dbscan:
                sil_dbscan = score
                best_eps, best_min = eps, min_samples
                best_labels_db = labels

print(f"Melhores params para DBSCAN: eps={best_eps}, min_samples={best_min}, Silhouette: {sil_dbscan:.3f}")




# Comparar resultados
if max(sil_k) > sil_dbscan:
    best_method = "K-Means"
    print("🏆 K-Means teve o melhor agrupamento com base no coeficiente de silhueta.")
else:
    best_method = "DBSCAN"
    print("🏆 DBSCAN teve o melhor agrupamento com base no coeficiente de silhueta.")


# Visualizar resultados
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette='tab10')
plt.title(f'K-Means (k={best_k})')

plt.subplot(1, 2, 2)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=best_labels_db, palette='tab10')
plt.title(f'DBSCAN (eps={best_eps}, min_samples={best_min})')

plt.tight_layout()
plt.show()
