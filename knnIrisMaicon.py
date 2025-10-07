import random
import math
import matplotlib.pyplot as plt

# Dados
K = 7  # Para avaliação k-NN, mas não usado no k-means
K_CLUSTERS = 3
MAX_ITER = 100

IRIS_SETOSA = 1
IRIS_VERSICOLOR = 2
IRIS_VIRGINICA = 3

databaseTreinamento=[
[5.4,3.7,1.5,0.2,IRIS_SETOSA],
[4.8,3.4,1.6,0.2,IRIS_SETOSA],
[4.8,3.0,1.4,0.1,IRIS_SETOSA],
[4.3,3.0,1.1,0.1,IRIS_SETOSA],
[5.8,4.0,1.2,0.2,IRIS_SETOSA],
[5.7,4.4,1.5,0.4,IRIS_SETOSA],
[5.4,3.9,1.3,0.4,IRIS_SETOSA],
[5.1,3.5,1.4,0.3,IRIS_SETOSA],
[5.7,3.8,1.7,0.3,IRIS_SETOSA],
[5.1,3.8,1.5,0.3,IRIS_SETOSA],
[5.4,3.4,1.7,0.2,IRIS_SETOSA],
[5.1,3.7,1.5,0.4,IRIS_SETOSA],
[4.6,3.6,1.0,0.2,IRIS_SETOSA],
[5.1,3.3,1.7,0.5,IRIS_SETOSA],
[4.8,3.4,1.9,0.2,IRIS_SETOSA],
[5.0,3.0,1.6,0.2,IRIS_SETOSA],
[5.0,3.4,1.6,0.4,IRIS_SETOSA],
[5.2,3.5,1.5,0.2,IRIS_SETOSA],
[5.2,3.4,1.4,0.2,IRIS_SETOSA],
[4.7,3.2,1.6,0.2,IRIS_SETOSA],
[4.8,3.1,1.6,0.2,IRIS_SETOSA],
[5.4,3.4,1.5,0.4,IRIS_SETOSA],
[5.2,4.1,1.5,0.1,IRIS_SETOSA],
[5.5,4.2,1.4,0.2,IRIS_SETOSA],
[4.9,3.1,1.5,0.1,IRIS_SETOSA],
[5.0,3.2,1.2,0.2,IRIS_SETOSA],
[5.5,3.5,1.3,0.2,IRIS_SETOSA],
[4.9,3.1,1.5,0.1,IRIS_SETOSA],
[4.4,3.0,1.3,0.2,IRIS_SETOSA],
[5.1,3.4,1.5,0.2,IRIS_SETOSA],
[5.0,3.5,1.3,0.3,IRIS_SETOSA],
[4.5,2.3,1.3,0.3,IRIS_SETOSA],
[4.4,3.2,1.3,0.2,IRIS_SETOSA],
[5.0,3.5,1.6,0.6,IRIS_SETOSA],
[5.1,3.8,1.9,0.4,IRIS_SETOSA],
[4.8,3.0,1.4,0.3,IRIS_SETOSA],
[5.1,3.8,1.6,0.2,IRIS_SETOSA],
[4.6,3.2,1.4,0.2,IRIS_SETOSA],
[5.3,3.7,1.5,0.2,IRIS_SETOSA],
[5.0,3.3,1.4,0.2,IRIS_SETOSA],
[5.0,2.0,3.5,1.0,IRIS_VERSICOLOR],
[5.9,3.0,4.2,1.5,IRIS_VERSICOLOR],
[6.0,2.2,4.0,1.0,IRIS_VERSICOLOR],
[6.1,2.9,4.7,1.4,IRIS_VERSICOLOR],
[5.6,2.9,3.6,1.3,IRIS_VERSICOLOR],
[6.7,3.1,4.4,1.4,IRIS_VERSICOLOR],
[5.6,3.0,4.5,1.5,IRIS_VERSICOLOR],
[5.8,2.7,4.1,1.0,IRIS_VERSICOLOR],
[6.2,2.2,4.5,1.5,IRIS_VERSICOLOR],
[5.6,2.5,3.9,1.1,IRIS_VERSICOLOR],
[5.9,3.2,4.8,1.8,IRIS_VERSICOLOR],
[6.1,2.8,4.0,1.3,IRIS_VERSICOLOR],
[6.3,2.5,4.9,1.5,IRIS_VERSICOLOR],
[6.1,2.8,4.7,1.2,IRIS_VERSICOLOR],
[6.4,2.9,4.3,1.3,IRIS_VERSICOLOR],
[6.6,3.0,4.4,1.4,IRIS_VERSICOLOR],
[6.8,2.8,4.8,1.4,IRIS_VERSICOLOR],
[6.7,3.0,5.0,1.7,IRIS_VERSICOLOR],
[6.0,2.9,4.5,1.5,IRIS_VERSICOLOR],
[5.7,2.6,3.5,1.0,IRIS_VERSICOLOR],
[5.5,2.4,3.8,1.1,IRIS_VERSICOLOR],
[5.5,2.4,3.7,1.0,IRIS_VERSICOLOR],
[5.8,2.7,3.9,1.2,IRIS_VERSICOLOR],
[6.0,2.7,5.1,1.6,IRIS_VERSICOLOR],
[5.4,3.0,4.5,1.5,IRIS_VERSICOLOR],
[6.0,3.4,4.5,1.6,IRIS_VERSICOLOR],
[6.7,3.1,4.7,1.5,IRIS_VERSICOLOR],
[6.3,2.3,4.4,1.3,IRIS_VERSICOLOR],
[5.6,3.0,4.1,1.3,IRIS_VERSICOLOR],
[5.5,2.5,4.0,1.3,IRIS_VERSICOLOR],
[5.5,2.6,4.4,1.2,IRIS_VERSICOLOR],
[6.1,3.0,4.6,1.4,IRIS_VERSICOLOR],
[5.8,2.6,4.0,1.2,IRIS_VERSICOLOR],
[5.0,2.3,3.3,1.0,IRIS_VERSICOLOR],
[5.6,2.7,4.2,1.3,IRIS_VERSICOLOR],
[5.7,3.0,4.2,1.2,IRIS_VERSICOLOR],
[5.7,2.9,4.2,1.3,IRIS_VERSICOLOR],
[6.2,2.9,4.3,1.3,IRIS_VERSICOLOR],
[5.1,2.5,3.0,1.1,IRIS_VERSICOLOR],
[5.7,2.8,4.1,1.3,IRIS_VERSICOLOR],
[6.5,3.2,5.1,2.0,IRIS_VIRGINICA],
[6.4,2.7,5.3,1.9,IRIS_VIRGINICA],
[6.8,3.0,5.5,2.1,IRIS_VIRGINICA],
[5.7,2.5,5.0,2.0,IRIS_VIRGINICA],
[5.8,2.8,5.1,2.4,IRIS_VIRGINICA],
[6.4,3.2,5.3,2.3,IRIS_VIRGINICA],
[6.5,3.0,5.5,1.8,IRIS_VIRGINICA],
[7.7,3.8,6.7,2.2,IRIS_VIRGINICA],
[7.7,2.6,6.9,2.3,IRIS_VIRGINICA],
[6.0,2.2,5.0,1.5,IRIS_VIRGINICA],
[6.9,3.2,5.7,2.3,IRIS_VIRGINICA],
[5.6,2.8,4.9,2.0,IRIS_VIRGINICA],
[7.7,2.8,6.7,2.0,IRIS_VIRGINICA],
[6.3,2.7,4.9,1.8,IRIS_VIRGINICA],
[6.7,3.3,5.7,2.1,IRIS_VIRGINICA],
[7.2,3.2,6.0,1.8,IRIS_VIRGINICA],
[6.2,2.8,4.8,1.8,IRIS_VIRGINICA],
[6.1,3.0,4.9,1.8,IRIS_VIRGINICA],
[6.4,2.8,5.6,2.1,IRIS_VIRGINICA],
[7.2,3.0,5.8,1.6,IRIS_VIRGINICA],
[7.4,2.8,6.1,1.9,IRIS_VIRGINICA],
[7.9,3.8,6.4,2.0,IRIS_VIRGINICA],
[6.4,2.8,5.6,2.2,IRIS_VIRGINICA],
[6.3,2.8,5.1,1.5,IRIS_VIRGINICA],
[6.1,2.6,5.6,1.4,IRIS_VIRGINICA],
[7.7,3.0,6.1,2.3,IRIS_VIRGINICA],
[6.3,3.4,5.6,2.4,IRIS_VIRGINICA],
[6.4,3.1,5.5,1.8,IRIS_VIRGINICA],
[6.0,3.0,4.8,1.8,IRIS_VIRGINICA],
[6.9,3.1,5.4,2.1,IRIS_VIRGINICA],
[6.7,3.1,5.6,2.4,IRIS_VIRGINICA],
[6.9,3.1,5.1,2.3,IRIS_VIRGINICA],
[5.8,2.7,5.1,1.9,IRIS_VIRGINICA],
[6.8,3.2,5.9,2.3,IRIS_VIRGINICA],
[6.7,3.3,5.7,2.5,IRIS_VIRGINICA],
[6.7,3.0,5.2,2.3,IRIS_VIRGINICA],
[6.3,2.5,5.0,1.9,IRIS_VIRGINICA],
[6.5,3.0,5.2,2.0,IRIS_VIRGINICA],
[6.2,3.4,5.4,2.3,IRIS_VIRGINICA],
[5.9,3.0,5.1,1.8,IRIS_VIRGINICA]
]

databaseTeste=[
[5.1,3.5,1.4,0.2,IRIS_SETOSA],
[4.9,3.0,1.4,0.2,IRIS_SETOSA],
[4.7,3.2,1.3,0.2,IRIS_SETOSA],
[4.6,3.1,1.5,0.2,IRIS_SETOSA],
[5.0,3.6,1.4,0.2,IRIS_SETOSA],
[5.4,3.9,1.7,0.4,IRIS_SETOSA],
[4.6,3.4,1.4,0.3,IRIS_SETOSA],
[5.0,3.4,1.5,0.2,IRIS_SETOSA],
[4.4,2.9,1.4,0.2,IRIS_SETOSA],
[4.9,3.1,1.5,0.1,IRIS_SETOSA],
[7.0,3.2,4.7,1.4,IRIS_VERSICOLOR],
[6.4,3.2,4.5,1.5,IRIS_VERSICOLOR],
[6.9,3.1,4.9,1.5,IRIS_VERSICOLOR],
[5.5,2.3,4.0,1.3,IRIS_VERSICOLOR],
[6.5,2.8,4.6,1.5,IRIS_VERSICOLOR],
[5.7,2.8,4.5,1.3,IRIS_VERSICOLOR],
[6.3,3.3,4.7,1.6,IRIS_VERSICOLOR],
[4.9,2.4,3.3,1.0,IRIS_VERSICOLOR],
[6.6,2.9,4.6,1.3,IRIS_VERSICOLOR],
[5.2,2.7,3.9,1.4,IRIS_VERSICOLOR],
[6.3,3.3,6.0,2.5,IRIS_VIRGINICA],
[5.8,2.7,5.1,1.9,IRIS_VIRGINICA],
[7.1,3.0,5.9,2.1,IRIS_VIRGINICA],
[6.3,2.9,5.6,1.8,IRIS_VIRGINICA],
[6.5,3.0,5.8,2.2,IRIS_VIRGINICA],
[7.6,3.0,6.6,2.1,IRIS_VIRGINICA],
[4.9,2.5,4.5,1.7,IRIS_VIRGINICA],
[7.3,2.9,6.3,1.8,IRIS_VIRGINICA],
[6.7,2.5,5.8,1.8,IRIS_VIRGINICA],
[7.2,3.6,6.1,2.5,IRIS_VIRGINICA]
]

# Funções auxiliares
def calcularDistanciaEuclidiana(a, b):
    return math.sqrt(sum((a[i] - b[i])**2 for i in range(4)))

def nomeClasse(classe):
    if classe == IRIS_SETOSA:
        return "Iris Setosa"
    if classe == IRIS_VERSICOLOR:
        return "Iris Versicolor"
    if classe == IRIS_VIRGINICA:
        return "Iris Virginica"

# --- K-MEANS ---
# Inicializa centróides aleatoriamente
centróides = random.sample([linha[:4] for linha in databaseTreinamento], K_CLUSTERS)

for it in range(MAX_ITER):
    clusters = {i: [] for i in range(K_CLUSTERS)}

    # Atribui cada ponto ao centróide mais próximo
    for ponto in databaseTreinamento:
        distancias = [calcularDistanciaEuclidiana(ponto[:4], c) for c in centróides]
        index_cluster = distancias.index(min(distancias))
        clusters[index_cluster].append(ponto)

    # Atualiza centróides
    novos_centróides = []
    for i in range(K_CLUSTERS):
        if clusters[i]:
            media = [sum(col)/len(col) for col in zip(*[p[:4] for p in clusters[i]])]
            novos_centróides.append(media)
        else:
            novos_centróides.append(random.choice([linha[:4] for linha in databaseTreinamento]))
    
    # Convergência
    if novos_centróides == centróides:
        break
    centróides = novos_centróides

# Mapear clusters para classe predominante
cluster_para_classe = {}
for i, pontos in clusters.items():
    contagem = {IRIS_SETOSA:0, IRIS_VERSICOLOR:0, IRIS_VIRGINICA:0}
    for p in pontos:
        contagem[p[4]] += 1
    cluster_para_classe[i] = max(contagem, key=contagem.get)

# --- AVALIAÇÃO ---
acertos = 0
pontos_por_cluster = {0: 0, 1: 0, 2: 0}

for instancia in databaseTeste:
    distancias = [calcularDistanciaEuclidiana(instancia[:4], c) for c in centróides]
    cluster_mais_proximo = distancias.index(min(distancias))
    pontos_por_cluster[cluster_mais_proximo] += 1
    classe_predita = cluster_para_classe[cluster_mais_proximo]
    print(f"Instância {instancia[:4]} -> {nomeClasse(classe_predita)} (Cluster {cluster_mais_proximo})")
    if classe_predita == instancia[4]:
        acertos += 1

print(f"\nPercentual de acerto: {(acertos/len(databaseTeste))*100:.2f}%")

print("\nQuantidade de pontos em cada cluster:")
for i, qtd in pontos_por_cluster.items():
    print(f"  Cluster {i}: {qtd} pontos")

import matplotlib.pyplot as plt

colors = ['r', 'g', 'b']

plt.figure(figsize=(12, 5))

# --- Gráfico 1: Clusters formados (treinamento) ---
plt.subplot(1, 2, 1)
for i, pontos in clusters.items():
    x = [p[0] for p in pontos]
    y = [p[1] for p in pontos]
    plt.scatter(x, y, color=colors[i], label=f'Cluster {i+1}')
plt.scatter([c[0] for c in centróides], [c[1] for c in centróides],
            color='k', marker='X', s=100, label='Centróides')
plt.xlabel('Comprimento da Sépala')
plt.ylabel('Largura da Sépala')
plt.title('Clusters (Treinamento)')
plt.legend()

# --- Gráfico 2: Banco de teste com predições ---
plt.subplot(1, 2, 2)
for instancia in databaseTeste:
    # Calcula o cluster mais próximo
    distancias = [calcularDistanciaEuclidiana(instancia[:4], c) for c in centróides]
    cluster_mais_proximo = distancias.index(min(distancias))
    classe_predita = cluster_para_classe[cluster_mais_proximo]

    # Plot do ponto de teste com a cor do cluster correspondente
    plt.scatter(instancia[0], instancia[1], color=colors[cluster_mais_proximo])

plt.scatter([c[0] for c in centróides], [c[1] for c in centróides],
            color='k', marker='X', s=100, label='Centróides')
plt.xlabel('Comprimento da Sépala')
plt.ylabel('Largura da Sépala')
plt.title('Banco de Teste (Classificado pelos Clusters)')
plt.legend()

plt.tight_layout()
plt.show()
