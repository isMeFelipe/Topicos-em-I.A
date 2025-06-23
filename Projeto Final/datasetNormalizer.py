import pandas as pd
from sklearn.preprocessing import LabelEncoder

# =======================================
# 1. Carregamento e limpeza inicial
# =======================================

df = pd.read_csv("okcupid_profiles.csv")

# Remover colunas de texto livre e completamente vazias
cols_to_drop = ['location', 'religion', 'sign', 'speaks', 'ethnicity', 'last_online', 'body_type']
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

essay_cols = [col for col in df.columns if col.startswith('essay')]
df.drop(columns=essay_cols, inplace=True)
df.dropna(axis=1, how='all', inplace=True)

# =======================================
# 2. Encoding de colunas categóricas simples
# =======================================

max_unique = 10
categorical_cols = df.select_dtypes(include=['object']).columns
few_cat_cols = [col for col in categorical_cols if df[col].nunique() <= max_unique and col != 'education' and col != 'orientation' and col != 'sex' and col != 'drinks' and col != 'smokes']

label_encoders = {}
for col in few_cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# =======================================
# 3. Discretização de idade
# =======================================

age_bins = [18, 23, 30, 40, 50, 60, 120]
age_labels = [0, 1, 2, 3, 4, 5]
df['age'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
df['age'] = df['age'].cat.add_categories([-1]).fillna(-1).astype(int)

# =======================================
# 4. Mapeamento de escolaridade
# =======================================

education_mapping = {
    # Ensino médio ou menos
    'high school': 0, 'dropped out of high school': 0, 'working on high school': 0, 'graduated from high school': 0,
    # Superior incompleto
    'college/university': 1, 'working on college/university': 1, 'dropped out of college/university': 1,
    'dropped out of two-year college': 1, 'two-year college': 1, 'working on two-year college': 1,
    # Superior completo
    'graduated from college/university': 2, 'graduated from two-year college': 2,
    # Pós-graduação
    'masters program': 3, 'working on masters program': 3, 'graduated from masters program': 3, 'dropped out of masters program': 3,
    # Doutorado / especializações
    'ph.d program': 4, 'working on ph.d program': 4, 'graduated from ph.d program': 4, 'dropped out of ph.d program': 4,
    # Profissionalizantes / específicos
    'law school': 5, 'working on law school': 5, 'graduated from law school': 5, 'dropped out of law school': 5,
    'med school': 5, 'working on med school': 5, 'graduated from med school': 5, 'dropped out of med school': 5,
    'space camp': 5, 'working on space camp': 5, 'graduated from space camp': 5, 'dropped out of space camp': 5
}
df['education'] = df['education'].map(education_mapping).fillna(-1).astype(int)

# =======================================
# 5. Mapeamento de dieta
# =======================================

diet_mapping = {
    # Livre
    'anything': 0, 'mostly anything': 0, 'strictly anything': 0,
    # Vegetariana/Vegana leve
    'vegetarian': 1, 'mostly vegetarian': 1, 'vegan': 1, 'mostly vegan': 1,
    # Vegetariana/Vegana estrita
    'strictly vegetarian': 2, 'strictly vegan': 2,
    # Religiosa leve
    'kosher': 3, 'halal': 3, 'mostly kosher': 3, 'mostly halal': 3,
    # Religiosa estrita
    'strictly kosher': 4, 'strictly halal': 4,
    # Outros
    'other': 5, 'mostly other': 5, 'strictly other': 6
}
df['diet'] = df['diet'].map(diet_mapping).fillna(-1).astype(int)

# =======================================
# 6. Mapeamento de profissão por área
# =======================================

job_mapping = {
    'artistic / musical / writer': 0, 'entertainment / media': 0,
    'banking / financial / real estate': 1, 'sales / marketing / biz dev': 1, 'executive / management': 1,
    'clerical / administrative': 2,
    'computer / hardware / software': 3, 'science / tech / engineering': 3,
    'construction / craftsmanship': 4, 'transportation': 4,
    'education / academia': 5, 'student': 5,
    'medicine / health': 6,
    'law / legal services': 7, 'political / government': 7,
    'hospitality / travel': 8,
    'military': 9,
    'other': 10, 'rather not say': 10, 'unemployed': 10, 'retired': 10
}
df['job'] = df['job'].map(job_mapping).fillna(-1).astype(int)

# =======================================
# 7. Processar 'offspring' em duas colunas
# =======================================

def parse_offspring(val):
    val = str(val)
    if 'has kids' in val:
        has_kids = 2
    elif 'has a kid' in val:
        has_kids = 1
    elif "doesn't have kids" in val:
        has_kids = 0
    else:
        has_kids = -1
    
    if "doesn't want" in val:
        wants_kids = 0
    elif 'might want' in val:
        wants_kids = 1
    elif 'wants' in val:
        wants_kids = 2
    else:
        wants_kids = -1

    return pd.Series([has_kids, wants_kids])

df[['has_kids', 'wants_kids']] = df['offspring'].apply(parse_offspring)
df.drop(columns=['offspring'], inplace=True)

# =======================================
# 8. Processar 'height' em valores discretizados
# =======================================

height_bins = [0, 62, 65, 68, 71, 74, float('inf')]
height_labels = [0, 1, 2, 3, 4, 5]  # 0=muito baixo, 1=baixo, 2=Médio-baixo, 3=Médio-alto, 4=Alto, 5=Muito alto

df['height_group'] = pd.cut(df['height'], bins=height_bins, labels=height_labels, right=False)
df['height_group'] = df['height_group'].cat.add_categories([-1]).fillna(-1).astype(int)
df = df.drop(columns=['height'])
df = df.rename(columns={'height_group': 'height'})

# =======================================
# 8. Processar 'pets' em valores discretizados
# =======================================

pets_affinity_mapping = {
    # 0 - Não gosta
    'dislikes cats': 0,
    'dislikes dogs': 0,
    'dislikes dogs and dislikes cats': 0,

    # 1 - Neutro
    'has cats': 1,
    'has dogs': 1,
    'has dogs and has cats': 1,
    'has dogs and dislikes cats': 1,
    'dislikes dogs and has cats': 1,

    # 2 - Gosta
    'likes cats': 2,
    'likes dogs': 2,
    'likes dogs and likes cats': 2,
    'likes dogs and has cats': 2,
    'has dogs and likes cats': 2,
    'dislikes dogs and likes cats': 2
}

df['pets_affinity'] = df['pets'].map(pets_affinity_mapping).fillna(-1).astype(int)
df = df.drop(columns=['pets'])
df = df.rename(columns={'pets_affinity': 'pets'})

# =======================================
#  9. Processar 'income' em valores discretizados
# =======================================

def map_income(value):
    if value == -1:
        return 0
    elif value <= 30000:
        return 1
    elif value <= 60000:
        return 2
    elif value <= 100000:
        return 3
    elif value <= 250000:
        return 4
    elif value <= 1000000:
        return 5
    else:
        return 6

df['income_grouped'] = df['income'].apply(map_income)
df = df.drop(columns=['income'])
df = df.rename(columns={'income_grouped': 'income'})


# =======================================
#  10. Orientation em valores discretizados
# =======================================

orientation_mapping = {'gay': 0, 'bisexual': 1, 'straight': 2}
df['orientation_group'] = df['orientation'].map(orientation_mapping).fillna(-1).astype(int)

df = df.drop(columns=['orientation'])
df = df.rename(columns={'orientation_group': 'orientation'})

# =======================================
# 11. Sex em valores discretizados
# =======================================

sex_mapping = {'m': 0, 'f': 1}
df['sex_group'] = df['sex'].map(sex_mapping).fillna(-1).astype(int)

df = df.drop(columns=['sex'])
df = df.rename(columns={'sex_group': 'sex'})

# =======================================
# 12. Agrupar 'drinks'
# =======================================

# Mapeamento para drinks
drinks_mapping = {
    'not at all': 0,
    'rarely': 1,
    'socially': 1,
    'often': 2,
    'very often': 2,
    'desperately': 2
}
df['drinks_grouped'] = df['drinks'].map(drinks_mapping).fillna(-1).astype(int)
df = df.drop(columns=['drinks'])
df = df.rename(columns={'drinks_grouped': 'drinks'})

# =======================================
# 13. Agrupar 'smokes'
# =======================================
smokes_mapping = {
    'no': 0,
    'trying to quit': 0,
    'sometimes': 1,
    'when drinking': 1,
    'yes': 1
}
df['smokes_grouped'] = df['smokes'].map(smokes_mapping).fillna(-1).astype(int)
df = df.drop(columns=['smokes'])
df = df.rename(columns={'smokes_grouped': 'smokes'})

# =======================================
# Salvar resultado
# =======================================

df.to_csv("okcupid_profiles_encoded.csv", index=False)
print("\n✅ Dataset salvo como 'okcupid_profiles_encoded.csv'")
