import pandas as pd

def listar_valores_unicos(df: pd.DataFrame, colunas):
    if isinstance(colunas, str):
        colunas = [colunas]
    
    resultado = {}
    for col in colunas:
        if col in df.columns:
            valores_unicos = df[col].dropna().unique()
            resultado[col] = sorted(valores_unicos.tolist())
        else:
            resultado[col] = None  # Coluna não existe
    return resultado

df = pd.read_csv("okcupid_profiles.csv")
valores = listar_valores_unicos(df, ['drinks','smokes'])
print(valores)
