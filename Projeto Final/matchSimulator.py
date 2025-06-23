import pandas as pd
import random
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

# === 2. Gerar pares ===
relacoes_set = set()
relacoes_final = []

match_count = 0
dismatch_count = 0

# 2.1 Gerar MATCHES (mesmo cluster + compatibilidade)
for cluster, members in clusters.items():
    combos = list(combinations(members, 2))
    random.shuffle(combos)

    for a, b in combos:
        if a == b:
            continue
        par = tuple(sorted((a, b)))
        if par in relacoes_set:
            continue

        ori1, sex1 = orientation_dict[a], sex_dict[a]
        ori2, sex2 = orientation_dict[b], sex_dict[b]

        if check_compatibility(ori1, sex1, ori2, sex2):
            relacoes_set.add(par)
            relacoes_final.append({
                'id1': a, 'sex1': sex1, 'orientation1': ori1,
                'id2': b, 'sex2': sex2, 'orientation2': ori2,
                'match': 1
            })
            match_count += 1
            if match_count >= NUM_MATCHES:
                break
    if match_count >= NUM_MATCHES:
        break

# 2.2 Gerar DISMATCHES (incompatíveis ou clusters diferentes)
while dismatch_count < NUM_DISMATCHES:
    a, b = random.sample(ids, 2)
    if a == b:
        continue
    par = tuple(sorted((a, b)))
    if par in relacoes_set:
        continue

    same_cluster = int(cluster_dict[a] == cluster_dict[b])
    ori1, sex1 = orientation_dict[a], sex_dict[a]
    ori2, sex2 = orientation_dict[b], sex_dict[b]
    compatible = check_compatibility(ori1, sex1, ori2, sex2)

    if not (same_cluster == 1 and compatible):
        relacoes_set.add(par)
        relacoes_final.append({
            'id1': a, 'sex1': sex1, 'orientation1': ori1,
            'id2': b, 'sex2': sex2, 'orientation2': ori2,
            'match': 0
        })
        dismatch_count += 1

# === 3. Salvar como CSV ===
df_relacoes = pd.DataFrame(relacoes_final)
df_relacoes.to_csv("relacionamentos_com_match.csv", index=False)
print("✅ Arquivo 'relacionamentos_com_match.csv' criado com sucesso!")
