import igraph as ig
import pandas as pd
import numpy as np
import random

## region[Preparação de consulta]
# Carregando a base de dados
try:
    df = pd.read_csv("Data_API.csv")
except Exception as e:
    print(f"Erro ao carregar o arquivo CSV: {e}")
    exit()

# Verifica se a coluna de data existe
if "Datetime_updated" not in df.columns:
    print("Erro: A coluna 'Datetime_updated' não foi encontrada no dataset.")
    exit()

# Converter a coluna 'Datetime_updated' para datetime
df["Datetime_updated"] = pd.to_datetime(df["Datetime_updated"], errors='coerce')

# Definir o ano desejado para o filtragem
target_year = 2020

# Filtrar apenas os dados do ano desejado
df = df[df["Datetime_updated"].dt.year == target_year]

if df.empty:
    print(f"Nenhuma transação encontrada para o ano {target_year}.")
    exit()

# Verifica se as colunas necessárias estão presentes
required_columns = {"Seller_address", "Buyer_address"}
if not required_columns.issubset(df.columns):
    print(f"Erro: O dataset não contém todas as colunas necessárias: {required_columns}")
    exit()

# Remove linhas com valores NaN nas colunas essenciais (para melhor qualidade dos dados)
df = df.dropna(subset=["Seller_address", "Buyer_address"])

# Garante que os endereços são strings
df["Seller_address"] = df["Seller_address"].astype(str)
df["Buyer_address"] = df["Buyer_address"].astype(str)


##endregion
##region[Criação do grafo]

edges = list(zip(df["Seller_address"], df["Buyer_address"]))
g = ig.Graph(directed=True)

vertices = set(df["Seller_address"].unique()).union(set(df["Buyer_address"].unique()))

g.add_vertices(list(vertices))
g.add_edges(edges)
##endregion

##region[Análise dos dados]
# Análise dos graus
in_degrees = g.degree(mode="in")  
out_degrees = g.degree(mode="out")  
total_degrees = g.degree(mode="all") 

print("Grau médio de entrada:", sum(in_degrees) / len(in_degrees))
print("Grau médio de saída:", sum(out_degrees) / len(out_degrees))
print("Grau médio total:", sum(total_degrees) / len(total_degrees))

# Otimização
sample_size = min(1000, len(g.vs))  # Limita a 1000 ou o total de vértices (otimização)
sample_vertices = random.sample(range(len(g.vs)), sample_size)

shortest_paths = g.shortest_paths(source=sample_vertices, target=sample_vertices)

finite_paths = np.array(shortest_paths).flatten()
finite_paths = finite_paths[np.isfinite(finite_paths)]
finite_paths = finite_paths[finite_paths > 0]

if len(finite_paths) > 0:
    min_shortest_path = np.min(finite_paths)
    max_shortest_path = np.max(finite_paths)
    mean_shortest_path = np.mean(finite_paths)

    print("Tamanho do menor caminho entre vértices:", min_shortest_path)
    print("Tamanho do maior caminho entre vértices:", max_shortest_path)
    print("Média do menor caminho entre vértices:", mean_shortest_path)
else:
    print("Não há caminhos válidos no grafo.")

# Componentes conectados
components = g.connected_components(mode="weak")

largest_component = max(components, key=len)

print("Número de componentes conectados:", len(components))
print("Tamanho do maior componente:", len(largest_component))
