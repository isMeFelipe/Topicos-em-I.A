import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# === Carregar dataset ===
df = pd.read_csv("okcupid_profiles_encoded.csv")

# === Selecionar top features ===
top_features = ['status', 'wants_kids', 'smokes', 'diet', 'income', 'has_kids', 'sex', 'drinks', 'height', 'age']
df_top = df[top_features].dropna().reset_index(drop=True)

# === Label Encoding para variáveis categóricas com poucas categorias ===
categorical_features = ['status', 'wants_kids', 'smokes', 'diet', 'has_kids', 'sex', 'drinks']
for col in categorical_features:
    le = LabelEncoder()
    df_top[col] = le.fit_transform(df_top[col])

# === Normalizar dados ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_top)

# === Avaliar Silhouette para k de 2 a 15 ===
silhouette_scores = []
k_values = range(2, 16)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=50)
    clusters = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, clusters)
    silhouette_scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Avaliação para escolha do número ideal de clusters')
plt.grid(True)
plt.show()

# === Escolher k e rodar KMeans final ===
N_CLUSTERS = int(input("Digite o número de clusters desejado (baseado no gráfico acima): "))
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=50)
clusters = kmeans.fit_predict(X_scaled)

# === Adicionar clusters ao df original e salvar ===
df_filtered = df.loc[df_top.index].copy()
df_filtered['cluster_group'] = clusters
df_filtered.to_csv("okcupid_profiles_clustered_top_features.csv", index=False)
print("\n✅ Clustering concluído! Arquivo salvo como 'okcupid_profiles_clustered_top_features.csv'")
