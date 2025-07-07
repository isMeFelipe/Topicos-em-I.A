import pandas as pd
import numpy as np
import joblib

# === 1. Carregar modelo e lista de features ===
modelo = joblib.load("random_forest_model.pkl")
feature_diff_names = joblib.load("model_features.pkl")

# === 2. Carregar dados dos perfis ===
df_profiles = pd.read_csv("okcupid_profiles_clustered_top_features.csv")

# Mesmas features usadas no treinamento
features = [
    'status', 'wants_kids', 'smokes', 'diet', 'income', 'has_kids',
    'drinks', 'height', 'age', 'drugs', 'education',
    'orientation', 'pets', 'job'
]

# === 3. Função para montar os dados de entrada com base em dois IDs ===
def construir_features_entrada(id1, id2):
    try:
        p1 = df_profiles[df_profiles['id'] == id1][features].values[0]
        p2 = df_profiles[df_profiles['id'] == id2][features].values[0]
    except IndexError:
        print("❌ Um ou ambos os IDs não foram encontrados.")
        return None

    diff = np.abs(p1 - p2)
    X_input = pd.DataFrame([diff], columns=feature_diff_names)
    return X_input

# === 4. Entrar com dois IDs manualmente ou de outro script ===
id1 = int(input("Digite o ID da primeira pessoa: "))
id2 = int(input("Digite o ID da segunda pessoa: "))

X_novo = construir_features_entrada(id1, id2)

if X_novo is not None:
    # === 5. Fazer predição ===
    pred = modelo.predict(X_novo)[0]
    prob = modelo.predict_proba(X_novo)[0][pred]

    print(f"\n🔍 Resultado da predição para o par ({id1}, {id2}):")
    if pred == 1:
        print(f"💘 Provável MATCH! (confiança: {prob:.2%})")
    else:
        print(f"🙅‍♂️ Provavelmente NÃO é um match. (confiança: {prob:.2%})")
