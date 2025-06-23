import pandas as pd
import random
import numpy as np
from itertools import combinations

# === 1. Carregar o dataset já clusterizado ===
df = pd.read_csv("okcupid_profiles_clustered_top_features.csv")

if 'id' not in df.columns:
    df = df.reset_index().rename(columns={'index': 'id'})

NUM_RELACIONAMENTOS = 1_000_000
NUM_MATCHES = NUM_RELACIONAMENTOS // 2
NUM_DISMATCHES = NUM_RELACIONAMENTOS - NUM_MATCHES

# Mapas auxiliares
cluster_dict = df.set_index('id')['cluster_group'].to_dict()
orientation_dict = df.set_index('id')['orientation'].to_dict()
sex_dict = df.set_index('id')['sex'].to_dict()
clusters = df.groupby('cluster_group')['id'].apply(list).to_dict()
ids = df['id'].tolist()
df_indexed = df.set_index('id')

# === Função de compatibilidade ===
def check_compatibility(ori1, sex1, ori2, sex2):
    if -1 in [ori1, sex1, ori2, sex2]:
        return False

    def match_one(o, s, o_other, s_other):
        if o == 0:  # gay
            return s == s_other and o_other in [0, 1]
        elif o == 2:  # straight
            return s != s_other and o_other in [2, 1]
        elif o == 1:  # bisexual
            if o_other == 0:
                return s == s_other
            elif o_other == 2:
                return s != s_other
            elif o_other == 1:
                return True
            else:
                return False
        else:
            return False

    return match_one(ori1, sex1, ori2, sex2) and match_one(ori2, sex2, ori1, sex1)

# === Similaridade de perfil ===
def profile_similarity_score(p1, p2):
    keys = ['age', 'income', 'height', 'wants_kids', 'smokes', 'drinks', 'diet']
    diffs = [abs(p1[k] - p2[k]) for k in keys if k in p1 and k in p2]
    return np.mean(diffs)

# === Equação de match final ===
def match_score(a, b):
    p1, p2 = df_indexed.loc[a], df_indexed.loc[b]
    diff_score = profile_similarity_score(p1, p2)
    max_possible_diff = 10  # ajuste conforme suas features

    base_score = max(0.0, 1.0 - (diff_score / max_possible_diff))

    # Penaliza se incompatíveis, mas não zera
    if not check_compatibility(p1['orientation'], p1['sex'], p2['orientation'], p2['sex']):
        return round(base_score * 0.3, 4)  # penaliza 70%
    else:
        return round(base_score, 4)


# === 2. Gerar pares ===
relacoes_set = set()
relacoes_final = []

match_count = 0
# 2.1 Gerar MATCHES com score > 0.7 (ajustável)
for cluster, members in clusters.items():
    combos = list(combinations(members, 2))
    random.shuffle(combos)

    for a, b in combos:
        if a == b:
            continue
        par = tuple(sorted((a, b)))
        if par in relacoes_set:
            continue

        score = match_score(a, b)
        if score > 0.7:
            relacoes_set.add(par)
            relacoes_final.append({
                'id1': a, 'sex1': sex_dict[a], 'orientation1': orientation_dict[a],
                'id2': b, 'sex2': sex_dict[b], 'orientation2': orientation_dict[b],
                'match': 1,
                'match_score': round(score, 4)
            })
            match_count += 1
            if match_count >= NUM_MATCHES:
                break
    if match_count >= NUM_MATCHES:
        break

# 2.2 Gerar DISMATCHES com score <= 0.3 (ou incompatíveis)
dismatch_count = 0
while dismatch_count < NUM_DISMATCHES:
    a, b = random.sample(ids, 2)
    if a == b:
        continue
    par = tuple(sorted((a, b)))
    if par in relacoes_set:
        continue

    score = match_score(a, b)
    if score <= 0.3:
        relacoes_set.add(par)
        relacoes_final.append({
            'id1': a, 'sex1': sex_dict[a], 'orientation1': orientation_dict[a],
            'id2': b, 'sex2': sex_dict[b], 'orientation2': orientation_dict[b],
            'match': 0,
            'match_score': round(score, 4)
        })
        dismatch_count += 1

# === 3. Salvar como CSV ===
df_relacoes = pd.DataFrame(relacoes_final)
df_relacoes.to_csv("relacionamentos_com_match.csv", index=False)
print("\n✅ Arquivo 'relacionamentos_com_match.csv' criado com sucesso!")
