import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import f_classif
import numpy as np

# === Carregar dataset ===
df = pd.read_csv("okcupid_profiles_encoded.csv")

# === Adicionar coluna 'id' única (se não existir) ===
if 'id' not in df.columns:
    df.insert(0, 'id', range(len(df)))

# === Selecionar top features ===
top_features = ['status', 'wants_kids', 'smokes', 'diet', 'income', 'has_kids',
                'drinks', 'height', 'age', 'orientation', 'pets']
df_top = df[top_features].dropna().reset_index(drop=True)

# === Label Encoding para variáveis categóricas ===
categorical_features = ['status', 'wants_kids', 'smokes', 'diet', 'has_kids', 'orientation', 'drinks']
for col in categorical_features:
    le = LabelEncoder()
    df_top[col] = le.fit_transform(df_top[col])

# === Normalizar dados ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_top)

# === Definir número fixo de clusters ===
N_CLUSTERS = 10  # Altere este valor conforme necessário
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=50)
clusters = kmeans.fit_predict(X_scaled)

# === Adicionar clusters ao DataFrame original ===
df_filtered = df.loc[df_top.index].copy()
df_filtered['cluster_group'] = clusters

# === Salvar resultado com ID e cluster ===
df_filtered.to_csv("okcupid_profiles_clustered_top_features.csv", index=False)
print("\n✅ Clustering concluído! Arquivo salvo como 'okcupid_profiles_clustered_top_features.csv'")

# === Calcular ANOVA para avaliação de variância entre clusters ===
f_values, p_values = f_classif(X_scaled, clusters)

anova_df = pd.DataFrame({
    'feature': top_features,
    'F_value': f_values,
    'p_value': p_values
}).sort_values(by='F_value', ascending=False)

print("\n=== ANOVA entre clusters ===")
print(anova_df.to_string(index=False))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# === Redução de dimensionalidade com PCA ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# === Criar DataFrame para visualização ===
df_pca = pd.DataFrame({
    'PCA1': X_pca[:, 0],
    'PCA2': X_pca[:, 1],
    'Cluster': clusters
})

# === Plotar clusters em 2D ===
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', s=60)
plt.title("Visualização dos Clusters com PCA")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
