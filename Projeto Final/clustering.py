import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# === Configuração ===
INPUT_FILE = "okcupid_profiles_encoded.csv"
OUTPUT_FILE = "okcupid_profiles_clustered.csv"
LIMIT_ROWS = None  # ➤ defina como None para usar todo o dataset

# === 1. Carregar o dataset ===
df = pd.read_csv(INPUT_FILE)

# Aplicar limite se definido
if LIMIT_ROWS is not None:
    df = df.sample(n=LIMIT_ROWS, random_state=42).reset_index(drop=True)

# === 2. Adicionar ID único ===
df.insert(0, 'id', range(1, len(df) + 1))

# === 3. Selecionar colunas para clustering ===
drop_cols = ['job', 'last_online', 'location', 'offspring', 'religion', 'sign', 'speaks']
df_clustering = df.drop(columns=[col for col in drop_cols if col in df.columns])

# === 4. Remover valores nulos ===
df_clustering = df_clustering.dropna()

# === 5. Selecionar dados numéricos e normalizar ===
X = df_clustering.select_dtypes(include=['int64', 'float64'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 6. Avaliar Silhouette Score para k de 2 a 10 ===
silhouette_scores = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(score)

# === 7. Plotar o gráfico do Silhouette Score ===
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Avaliação para escolha do número ideal de clusters')
plt.grid(True)
plt.show()

# === 8. Escolher o número de clusters com input ===
N_CLUSTERS = int(input("Digite o número de clusters desejado (baseado no gráfico acima): "))

# === 9. Rodar K-Means com o k escolhido ===
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# === 10. Adicionar coluna de cluster ===
df_filtered = df.loc[X.index].copy()
df_filtered['cluster_group'] = clusters

# === 11. Salvar novo dataset ===
df_filtered.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Clustering concluído! Arquivo salvo como: {OUTPUT_FILE}")

# === 12. Visualização com PCA ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plot_df = pd.DataFrame({
    'PCA1': X_pca[:, 0],
    'PCA2': X_pca[:, 1],
    'cluster': clusters
})

plt.figure(figsize=(10, 6))
sns.scatterplot(data=plot_df, x='PCA1', y='PCA2', hue='cluster', palette='Set2', s=60)
plt.title(f"Visualização dos Clusters (K={N_CLUSTERS} + PCA)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()
