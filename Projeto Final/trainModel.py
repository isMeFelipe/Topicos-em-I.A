import pandas as pd
import numpy as np
import joblib  # <-- Novo
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# === 1. Carregar dados ===
df_profiles = pd.read_csv("okcupid_profiles_clustered_top_features.csv")
df_matches = pd.read_csv("relacionamentos_com_match.csv")

# Features que serão usadas no modelo
features = [
    'status', 'wants_kids', 'smokes', 'diet', 'income', 'has_kids',
    'drinks', 'height', 'age', 'drugs', 'education',
    'orientation', 'pets', 'job'
]

# === 2. Criar features de comparação entre os pares ===
def build_feature_pairs(row):
    p1 = df_profiles[df_profiles['id'] == row['id1']][features].values[0]
    p2 = df_profiles[df_profiles['id'] == row['id2']][features].values[0]
    return np.abs(p1 - p2)  # Diferença absoluta

# Criar dataset de treino
X = np.vstack(df_matches.apply(build_feature_pairs, axis=1))
y = df_matches['match'].values

# Nome das features para referência
feature_diff_names = [f'diff_{f}' for f in features]
df_train = pd.DataFrame(X, columns=feature_diff_names)
df_train['match'] = y

# === 3. Separar treino e teste ===
X = df_train.drop(columns=['match'])
y = df_train['match']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 4. Treinar modelo Random Forest ===
rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

# === 5. Avaliação ===
y_pred = rf.predict(X_test)
print("=== Classification Report (Random Forest) ===")
print(classification_report(y_test, y_pred))
print("\n=== Matriz de Confusão ===")
print(confusion_matrix(y_test, y_pred))

# === 6. Importância das features ===
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("\n=== Importância das Features ===")
for idx in sorted_idx:
    print(f"{X.columns[idx]}: {importances[idx]:.4f}")

# === 7. Visualização ===
plt.figure(figsize=(10, 6))
plt.barh([X.columns[i] for i in sorted_idx][::-1], importances[sorted_idx][::-1])
plt.xlabel("Importância")
plt.title("Importância das Features - Random Forest")
plt.tight_layout()
plt.show()

# === 8. Salvar o modelo treinado ===
joblib.dump(rf, "random_forest_model.pkl")
joblib.dump(feature_diff_names, "model_features.pkl")
print("\n✅ Modelo salvo como 'random_forest_model.pkl'")
print("✅ Lista de features salva como 'model_features.pkl'")
