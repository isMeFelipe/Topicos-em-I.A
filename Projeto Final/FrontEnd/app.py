from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__, template_folder="templates", static_folder="static")

# Caminhos dos arquivos
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ORIGINAL_DATASET_PATH = os.path.join(BASE_DIR, "okcupid_profiles_clustered_top_features.csv")
CURRENT_DATASET_PATH = os.path.join(BASE_DIR, "current_profiles_data.csv")
MATCHES_PATH = os.path.join(BASE_DIR, "precalculated_matches.csv")
MODEL_PATH = os.path.join(BASE_DIR, "random_forest_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model_features.pkl")

# Garantir que o dataset atual existe
if not os.path.exists(CURRENT_DATASET_PATH):
    pd.read_csv(ORIGINAL_DATASET_PATH).to_csv(CURRENT_DATASET_PATH, index=False)

# Carregar modelo e lista de features
modelo = joblib.load(MODEL_PATH)
feature_diff_names = joblib.load(FEATURES_PATH)
features_modelo = feature_diff_names


# Features utilizadas no modelo
features = [
    'status', 'wants_kids', 'smokes', 'diet', 'income', 'has_kids',
    'drinks', 'height', 'age', 'orientation', 'pets', 'job',
    'education', 'drugs', 'sexo'
]

# Mapeamentos dos valores
mapeamentos = {
    "status": ["solteiro(a)", "namorando", "casado(a)"],
    "wants_kids": ["não quer", "talvez quer", "quer"],
    "smokes": ["não", "sim"],
    "diet": ["livre", "vegetariana/vegana leve", "vegetariana/vegana estrita", "religiosa leve", "religiosa estrita", "outros", "outros estrito"],
    "income": ["sem renda", "até 30k", "até 60k", "até 100k", "até 250k", "até 1M", "mais de 1M"],
    "has_kids": ["não tem filhos", "tem 1 filho", "tem filhos"],
    "drinks": ["nunca", "socialmente", "frequentemente"],
    "height": ["muito baixo", "baixo", "médio-baixo", "médio-alto", "alto", "muito alto"],
    "age": ["18–22", "23–29", "30–39", "40–49", "50–59", "60+"],
    "orientation": ["gay", "bissexual", "hétero"],
    "pets": ["não gosta", "neutro", "gosta"],
    "job": ["arte", "finanças", "admin", "tecnologia", "engenharia", "educação", "saúde", "direito/governo", "turismo", "outros"],
    "education": ["ensino médio ou menos", "superior incompleto", "superior completo", "mestrado", "doutorado", "profissionalizante"],
    "drugs": ["não", "às vezes", "frequentemente"],
    "sexo": ["masculino", "feminino", "outro"]
}

# Rótulos em português
rotulos_portugues = {
    "status": "Status",
    "wants_kids": "Deseja ter filhos",
    "smokes": "Fuma?",
    "diet": "Dieta",
    "income": "Renda",
    "has_kids": "Tem filhos?",
    "drinks": "Bebe?",
    "height": "Altura",
    "age": "Idade",
    "orientation": "Orientação sexual",
    "pets": "Gosta de animais?",
    "job": "Profissão",
    "education": "Escolaridade",
    "drugs": "Usa drogas?",
    "sexo": "Sexo"
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/criar_perfil", methods=["GET", "POST"])
def criar_perfil():
    if request.method == "POST":
        try:
            novo = {k: int(request.form[k]) for k in features if k in request.form}
            df = pd.read_csv(CURRENT_DATASET_PATH)
            novo['id'] = int(df['id'].max() + 1) if not df.empty else 1
            novo['cluster_group'] = -1
            df = pd.concat([df, pd.DataFrame([novo])], ignore_index=True)
            df.to_csv(CURRENT_DATASET_PATH, index=False)

            # Invalida matches que envolvem o novo perfil
            if os.path.exists(MATCHES_PATH):
                matches_df = pd.read_csv(MATCHES_PATH)
                matches_df = matches_df[(matches_df['profile_id'] != novo['id']) & (matches_df['matched_profile_id'] != novo['id'])]
                matches_df.to_csv(MATCHES_PATH, index=False)

            return redirect(url_for('ver_matches', user_id=novo['id']))
        except Exception as e:
            print("Erro ao criar perfil:", e)
            return "Erro ao criar perfil. Verifique os campos.", 400

    return render_template("criar_perfil.html", mapeamentos=mapeamentos, rotulos=rotulos_portugues)

@app.route("/ver_matches", methods=["GET"])
def ver_matches():
    user_id = request.args.get("user_id", type=int)
    page = request.args.get("page", default=1, type=int)
    df = pd.read_csv(CURRENT_DATASET_PATH)

    if user_id not in df['id'].values:
        return f"ID {user_id} não encontrado."

    # Features usadas pelo modelo (sem 'sexo')
    features_modelo = [f for f in features if f != 'sexo']

    if os.path.exists(MATCHES_PATH):
        matches_df = pd.read_csv(MATCHES_PATH)
        top_matches = matches_df[matches_df['profile_id'] == user_id]
    else:
        top_matches = pd.DataFrame()

    if top_matches.empty:
        base = df[df['id'] == user_id][features_modelo].values[0]
        outros = df[df['id'] != user_id]
        resultados = []

        for _, row in outros.iterrows():
            comparado = row[features_modelo].values
            diff = np.abs(base - comparado)

            if len(diff) == len(feature_diff_names):
                entrada = pd.DataFrame([diff], columns=feature_diff_names)
                prob = modelo.predict_proba(entrada)[0][1]
                resultados.append({
                    "profile_id": user_id,
                    "matched_profile_id": row['id'],
                    "match_probability": prob
                })

        # Ordenar os top 5 por maior probabilidade
        resultados = sorted(resultados, key=lambda x: x['match_probability'], reverse=True)
        top5 = resultados[:5]
        top_matches = pd.DataFrame(top5)
        top_matches['rank'] = range(1, len(top_matches) + 1)

        # IDs dos top 5
        matched_ids = top_matches['matched_profile_id'].tolist()

        # Candidatos possíveis para o match extra
        possiveis = df[
            (df['id'] != user_id) &
            (~df['id'].isin(matched_ids)) &
            (df['orientation'] == df[df['id'] == user_id]['orientation'].values[0]) &
            (abs(df['age'] - df[df['id'] == user_id]['age'].values[0]) <= 1)
        ]

        # Seleção do match aleatório entre 30–70%
        if not possiveis.empty:
            np.random.seed(user_id)
            candidatos = []
            for _, row in possiveis.iterrows():
                comparado = row[features_modelo].values
                diff = np.abs(base - comparado)
                entrada = pd.DataFrame([diff], columns=feature_diff_names)
                prob = modelo.predict_proba(entrada)[0][1]
                if 0.3 <= prob <= 0.7:
                    candidatos.append((row['id'], prob))

            if candidatos:
                escolhido = candidatos[np.random.randint(len(candidatos))]
                match_extra = {
                    "profile_id": user_id,
                    "matched_profile_id": escolhido[0],
                    "match_probability": escolhido[1],
                    "rank": 6
                }
                top_matches = pd.concat([top_matches, pd.DataFrame([match_extra])], ignore_index=True)

        # Salvar os matches no cache
        if os.path.exists(MATCHES_PATH):
            all_matches = pd.read_csv(MATCHES_PATH)
            all_matches = pd.concat([all_matches, top_matches], ignore_index=True)
        else:
            all_matches = top_matches
        all_matches.to_csv(MATCHES_PATH, index=False)

    # Paginação (fixo com 1 página)
    per_page = 5
    total_pages = 1

    top_ids = top_matches.sort_values("rank")["matched_profile_id"].tolist()
    df_filtered = df[df['id'].isin(top_ids)]
    df_filtered = df_filtered.set_index('id').loc[top_ids].reset_index()

    resultados = []
    for idx, row in df_filtered.iterrows():
        prob = top_matches[top_matches['matched_profile_id'] == row['id']]['match_probability'].values[0]
        atributos_legiveis = {
            f: (
                mapeamentos[f][int(row[f])]
                if pd.notnull(row[f]) and int(row[f]) in range(len(mapeamentos[f]))
                else "N/A"
            )
            for f in features  # Inclui 'sexo' para exibição
        }
        resultados.append({
            "id": row['id'],
            "prob": f"{prob * 100:.2f}%",
            "features": atributos_legiveis
        })

    return render_template("ver_matches.html", user_id=user_id, matches=resultados, page=page, total_pages=total_pages, rotulos=rotulos_portugues)

if __name__ == "__main__":
    app.run(debug=True)
