import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# 1. Carregar o dataset
df = pd.read_csv("okcupid_profiles.csv")

# Remover colunas que começam com 'essay' (texto livre)
essay_cols = [col for col in df.columns if col.startswith('essay')]
df = df.drop(columns=essay_cols)

# Remover colunas completamente vazias
df = df.dropna(axis=1, how='all')

# 2. Definir o limite de valores únicos para considerar como "poucas opções"
max_unique = 10

# 3. Selecionar colunas categóricas com poucas categorias (exceto education)
categorical_cols = df.select_dtypes(include=['object']).columns
few_cat_cols = [col for col in categorical_cols if df[col].nunique() <= max_unique and col != 'education']

# 4. Aplicar Label Encoding nas colunas selecionadas
label_encoders = {}
for col in few_cat_cols:
    le = LabelEncoder()
    df[col] = df[col].astype(str)  # Evita erro com valores nulos ou float
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 5. Discretizar a coluna age em faixas etárias (bins)
bins = [18, 23, 30, 40, 50, 60, 120]  # intervalos das idades
labels = [0, 1, 2, 3, 4, 5]            # rótulos para as faixas

df['age_binned'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
df['age_binned'] = df['age_binned'].cat.add_categories([-1]).fillna(-1).astype(int)  
# -1 indica idade fora do intervalo ou NaN

df = df.drop(columns=['age'])
df = df.rename(columns={'age_binned':'age'})

# 7. Mapear education para valores inteiros (clusterização)
education_mapping = {
    # Ensino médio ou menos
    'high school': 0,
    'dropped out of high school': 0,
    'working on high school': 0,
    'graduated from high school': 0,
    
    # Ensino superior incompleto
    'college/university': 1,
    'working on college/university': 1,
    'dropped out of college/university': 1,
    'dropped out of two-year college': 1,
    'two-year college': 1,
    'working on two-year college': 1,
    
    # Ensino superior completo
    'graduated from college/university': 2,
    'graduated from two-year college': 2,
    
    # Pós-graduação
    'masters program': 3,
    'working on masters program': 3,
    'graduated from masters program': 3,
    'dropped out of masters program': 3,
    
    # Doutorado / especializações
    'ph.d program': 4,
    'working on ph.d program': 4,
    'graduated from ph.d program': 4,
    'dropped out of ph.d program': 4,
    
    # Cursos específicos ou outros
    'law school': 5,
    'working on law school': 5,
    'graduated from law school': 5,
    'dropped out of law school': 5,
    'med school': 5,
    'working on med school': 5,
    'graduated from med school': 5,
    'dropped out of med school': 5,
    'space camp': 5,
    'working on space camp': 5,
    'graduated from space camp': 5,
    'dropped out of space camp': 5
}

# Aplicar o mapeamento, preenchendo com -1 onde não reconhecido ou NaN
df['education_mapped'] = df['education'].map(education_mapping).fillna(-1).astype(int)

# Remover a coluna original e renomear a nova
df = df.drop(columns=['education'])
df = df.rename(columns={'education_mapped': 'education'})


# 9. Salvar dataset processado
df.to_csv("okcupid_profiles_encoded.csv", index=False)
print("\nDataset salvo como 'okcupid_profiles_encoded.csv'")
