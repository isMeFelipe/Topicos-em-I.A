import pandas as pd
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import leidenalg


def visualizar_karate():
    G = ig.Graph.Famous("Zachary")
    layout = G.layout("fr")
    print(f"Número de vértices (Karate Club): {G.vcount()}")

    plt.figure(figsize=(10, 8))
    ig.plot(G, layout=layout, vertex_size=30, edge_color='gray', bbox=(800, 800), target=plt.gca())
    plt.title("Grafo Original do Clube de Caratê")
    plt.axis("off")
    plt.show()


def karate_edge_betweenness():
    G = ig.Graph.Famous("Zachary")
    layout = G.layout("fr")

    edge_bet = G.edge_betweenness()
    edges_sorted = sorted(enumerate(edge_bet), key=lambda x: x[1], reverse=True)

    print("Top 5 arestas por Edge Betweenness (Karate Club):")
    for i in range(5):
        source, target = G.es[edges_sorted[i][0]].tuple
        print(f"Aresta ({source}, {target}) com centralidade {edges_sorted[i][1]:.4f}")

    max_bet = max(edge_bet) if edge_bet else 1
    edge_widths = [eb / max_bet * 4 for eb in edge_bet]

    plt.figure(figsize=(10, 8))
    ig.plot(G, layout=layout, edge_width=edge_widths, edge_color="red", vertex_size=30, bbox=(800, 800), target=plt.gca())
    plt.title("Grafo do Clube de Caratê com Edge Betweenness")
    plt.axis("off")
    plt.show()


def karate_comunidades():
    G = ig.Graph.Famous("Zachary")
    layout = G.layout("fr")
    partition = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition)

    print(f"Número de comunidades detectadas (Karate): {len(partition)}")

    color_map = [community for community in partition.membership]
    plt.figure(figsize=(10, 8))
    ig.plot(G, layout=layout, vertex_color=color_map, vertex_size=30, bbox=(800, 800), target=plt.gca())
    plt.title("Grafo do Clube de Caratê com Comunidades (Leiden)")
    plt.axis("off")
    plt.show()


def carregar_dados_nft():
    df = pd.read_csv("Data_API.csv", low_memory=False)
    df["Datetime_updated"] = pd.to_datetime(df["Datetime_updated"], errors='coerce')
    df = df[df["Datetime_updated"].dt.year == 2020]
    df = df.dropna(subset=["Seller_address", "Buyer_address"])
    df["Seller_address"] = df["Seller_address"].astype(str)
    df["Buyer_address"] = df["Buyer_address"].astype(str)
    return df


def grafo_nft_original(df):
    edges = list(zip(df["Seller_address"], df["Buyer_address"]))
    vertices = list(set(df["Seller_address"]).union(set(df["Buyer_address"])))

    g_nft = ig.Graph(directed=True)
    g_nft.add_vertices(vertices)
    g_nft.add_edges(edges)

    print(f"Número de vértices (NFT): {g_nft.vcount()}, Número de arestas: {g_nft.ecount()}")

    layout = g_nft.layout("fr")
    plt.figure(figsize=(12, 10))
    ig.plot(g_nft, layout=layout, vertex_size=6, bbox=(800, 800), target=plt.gca())
    plt.title("Grafo NFT Original")
    plt.axis("off")
    plt.show()


def grafo_nft_edge_betweenness(df):
    print("Calculando edge betweenness com igraph...")

    edges = list(zip(df["Seller_address"], df["Buyer_address"]))
    vertices = list(set(df["Seller_address"]).union(set(df["Buyer_address"])))

    g_nft = ig.Graph(directed=True)
    g_nft.add_vertices(vertices)
    g_nft.add_edges(edges)

    print(f"Número de vértices (NFT): {g_nft.vcount()}, Número de arestas: {g_nft.ecount()}")

    edge_bet = g_nft.edge_betweenness()
    edges_sorted = sorted(enumerate(edge_bet), key=lambda x: x[1], reverse=True)

    print("Top 5 arestas por Edge Betweenness (NFT):")
    for i in range(5):
        source, target = g_nft.es[edges_sorted[i][0]].tuple
        print(f"Aresta ({source}, {target}) com centralidade {edges_sorted[i][1]:.4f}")

    max_bet = max(edge_bet) if edge_bet else 1
    edge_widths = [eb / max_bet * 4 for eb in edge_bet]

    layout = g_nft.layout("fr")
    plt.figure(figsize=(12, 10))
    ig.plot(
        g_nft,
        layout=layout,
        edge_width=edge_widths,
        edge_color="red",
        vertex_size=6,
        bbox=(800, 800),
        target=plt.gca()
    )
    plt.title("NFT - Edge Betweenness com iGraph")
    plt.axis("off")
    plt.show()


def grafo_nft_comunidades(df):
    edges = list(zip(df["Seller_address"], df["Buyer_address"]))
    vertices = list(set(df["Seller_address"]).union(set(df["Buyer_address"])))

    g_nft = ig.Graph(directed=True)
    g_nft.add_vertices(vertices)
    g_nft.add_edges(edges)

    print(f"Número de vértices (NFT): {g_nft.vcount()}, Número de arestas: {g_nft.ecount()}")

    partition = leidenalg.find_partition(g_nft, leidenalg.ModularityVertexPartition)

    print(f"Número de comunidades detectadas (NFT): {len(partition)}")

    color_map = [community for community in partition.membership]
    layout = g_nft.layout("fr")
    plt.figure(figsize=(12, 10))
    ig.plot(g_nft, layout=layout, vertex_color=color_map, vertex_size=6, bbox=(800, 800), target=plt.gca())
    plt.title("NFT - Comunidades (Leiden)")
    plt.axis("off")
    plt.show()


def menu():
    df_nft = carregar_dados_nft()

    while True:
        print("\n--- MENU ---")
        print("1 - Ver Grafo Karate Original")
        print("2 - Ver Karate com Edge Betweenness")
        print("3 - Ver Karate com Comunidades (Leiden)")
        print("4 - Ver NFT Original")
        print("5 - Ver NFT com Edge Betweenness")
        print("6 - Ver NFT com Comunidades (Leiden)")
        print("0 - Sair")
        opcao = input("Escolha uma opção: ")

        if opcao == "1":
            visualizar_karate()
        elif opcao == "2":
            karate_edge_betweenness()
        elif opcao == "3":
            karate_comunidades()
        elif opcao == "4":
            grafo_nft_original(df_nft)
        elif opcao == "5":
            grafo_nft_edge_betweenness(df_nft)
        elif opcao == "6":
            grafo_nft_comunidades(df_nft)
        elif opcao == "0":
            break
        else:
            print("Opção inválida!")


if __name__ == "__main__":
    menu()
