import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Carregar dataset ===
df = pd.read_csv("okcupid_profiles_encoded.csv")

# === 2. Filtrar colunas para clustering ===
drop_cols = ['last_online', 'location', 'religion', 'sign', 'speaks', 'ethnicity', 'body_type']
df_clustering = df.drop(columns=[col for col in drop_cols if col in df.columns])

# === 3. Remover linhas com valores nulos ===
df_clustering = df_clustering.dropna().reset_index(drop=True)

# === 4. Separar colunas numéricas e categóricas ===
num_cols = df_clustering.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df_clustering.select_dtypes(include=['object']).columns.tolist()

# === 5. Aplicar One-Hot Encoding para categóricas ===
if len(cat_cols) > 0:
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    cat_encoded = ohe.fit_transform(df_clustering[cat_cols])
    cat_encoded_df = pd.DataFrame(cat_encoded, columns=ohe.get_feature_names_out(cat_cols))
    df_clustering = df_clustering.drop(columns=cat_cols)
    df_clustering = pd.concat([df_clustering.reset_index(drop=True), cat_encoded_df.reset_index(drop=True)], axis=1)

# === 6. Normalizar dados numéricos e codificados ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clustering)

# === 7. Rodar KMeans para clustering ===
N_CLUSTERS = int(input("Digite o número de clusters desejado: "))
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=50)
clusters = kmeans.fit_predict(X_scaled)

# === 8. Análise de importância das features (ANOVA f_classif) ===
F_values, p_values = f_classif(X_scaled, clusters)
feature_names = df_clustering.columns

feat_importance = pd.DataFrame({
    'feature': feature_names,
    'F_value': F_values,
    'p_value': p_values
}).sort_values(by='F_value', ascending=False)

print("\n===== Features mais importantes para diferenciar os clusters =====\n")
print(feat_importance.head(15))

# === 9. Salvar resultado com cluster no dataframe original ===
df_filtered = df.loc[df_clustering.index].copy()
df_filtered['cluster_group'] = clusters
df_filtered.to_csv("okcupid_profiles_clustered.csv", index=False)
print("\n✅ Clustering concluído! Arquivo salvo como 'okcupid_profiles_clustered.csv'")

# === 10. Visualização PCA dos clusters para efeito visual ===
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
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
