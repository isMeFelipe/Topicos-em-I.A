import networkx as nx
import matplotlib.pyplot as plt
import igraph as ig
import leidenalg
import numpy as np

G = nx.karate_club_graph()
pos = nx.spring_layout(G, seed=42)

'''
    Grafo original
'''
plt.figure(figsize=(10, 8))
nx.draw(
    G, pos, with_labels=True,
    node_color='lightblue',
    edge_color='gray',
    node_size=600,
    font_size=10
)
plt.title("Grafo Original do Clube de Caratê")
plt.show()

#region[Edge Betweness]
'''
    Cálculo do Edge Betweeness
'''
edge_betweenness = nx.edge_betweenness_centrality(G)
edges_sorted = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)

print("Top 5 arestas por Edge Betweenness:")
for i in range(5):
    print(f"Aresta {edges_sorted[i][0]} com centralidade {edges_sorted[i][1]:.4f}")

edge_widths = [edge_betweenness[edge] * 30 for edge in G.edges()]

'''
    Grafo com destaque das arestas centrais
'''
plt.figure(figsize=(10, 8))
nx.draw(
    G, pos, with_labels=True,
    edge_color='red',
    width=edge_widths,
    node_color='lightblue',
    node_size=600,
    font_size=10
)
plt.title("Grafo do Clube de Caratê com Edge Betweenness")
plt.show()

#endregion

#region[Louvain]

'''
    Conversão para igraph
'''
G_ig = ig.Graph(edges=list(G.edges()))
G_ig.vs["name"] = list(G.nodes())

'''
    community_multilevel (Louvain)
'''
louvain_partition = G_ig.community_multilevel()
louvain_membership = louvain_partition.membership

print("\n[Louvain] Número de comunidades encontradas:", len(set(louvain_membership)))

num_comunidades = len(set(louvain_membership))
cmap = plt.colormaps["Set3"].resampled(num_comunidades)
colors_louvain = [cmap(i) for i in louvain_membership]

plt.figure(figsize=(10, 8))
nx.draw(
    G, pos, with_labels=True,
    node_color=colors_louvain,
    edge_color='gray',
    node_size=600,
    font_size=10
)
plt.title("Detecção de Comunidades com Louvain (Multilevel)")
plt.show()


#endregion
#region[Leiden]

'''
    community_leiden
'''
leiden_partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition)
leiden_membership = leiden_partition.membership

print("\n[Leiden] Número de comunidades encontradas:", len(set(leiden_membership)))

num_comunidades = len(set(leiden_membership))
cmap = plt.colormaps["Set1"].resampled(num_comunidades)
colors_leiden = [cmap(i) for i in leiden_membership]

plt.figure(figsize=(10, 8))
nx.draw(
    G, pos, with_labels=True,
    node_color=colors_leiden,
    edge_color='gray',
    node_size=600,
    font_size=10
)
plt.title("Detecção de Comunidades com Leiden")
plt.show()

#endregion
