import pandas as pd
import networkx as nx
from community import community_louvain
import matplotlib.pyplot as plt
import numpy as np

# 读取 CSV 文件football
df = pd.read_csv('data/Graph1.csv')

# 创建有向图
G = nx.Graph()

def force_directed_layout(graph, iterations=100, repulsive_force=0.1, attractive_force=0.1, k_coefficient=0.1,
                            gravity_coefficient=0.7, temperature_coefficient=0.1):
  # Initialize node positions on a circular layout
  num_nodes = len(graph.nodes())
  theta = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
  pos = {node: np.random.rand(2) for node in graph.nodes()}

  # Initial temperature
  temperature = 1.0

  # Perform iterations
  for iteration in range(iterations):
      # Compute repulsive forces based on node degree
      repulsive_forces = {node: np.zeros(2) for node in graph.nodes()}
      for u in graph.nodes():
        for v in graph.nodes():
          if u != v:
            delta = pos[u] - pos[v]
            distance = np.linalg.norm(delta)  # Euclidean distance
            degree_factor = graph.degree(u) * graph.degree(v)  # Product of degrees
            #repulsive_forces[u] += ((k_coefficient ** 2) *np.sqrt(degree_factor) * delta)  / distance   # Adjust the power or use a custom function
            repulsive_forces[u] += ((k_coefficient ** 2) * np.sqrt(degree_factor) * delta) / distance

      # Compute attractive forces based on edge weights
      attractive_forces = {node: np.zeros(2) for node in graph.nodes()}
      for u, v, weight in graph.edges(data='weight'):
        delta = pos[u] - pos[v]
        distance = np.linalg.norm(delta)  # Euclidean distance
        attractive_forces[u] +=k_coefficient * np.sqrt(distance * weight)
        attractive_forces[u] -=k_coefficient * np.sqrt(distance * weight)
      # Compute gravitational forces based on node degrees
      gravitational_forces = {node: np.zeros(2) for node in graph.nodes()}
      for node in graph.nodes():
        gravitational_forces[node] += gravity_coefficient * graph.degree(node)  # Adjust the power or use a custom function
      # Update node positions based on forces
      for node in graph.nodes():
        pos[node] += (repulsive_force * repulsive_forces[node] + attractive_force * attractive_forces[node] +gravitational_forces[node]) * temperature

       # Cool the system (reduce temperature)
      temperature *= 1 - temperature_coefficient

# 添加边和权重
for row in df.itertuples(index=False):
    G.add_edge(row[0], row[1], weight=row[2])
# 计算节点的 PageRank 值
pagerank = nx.pagerank(G)

# 根据 PageRank 值排序，选择前 15% 的节点作为种子节点
seed_nodes = sorted(pagerank, key=pagerank.get, reverse=True)[:int(len(G) * 0.12)]

# 将有向图转换为无向图
G_undirected = G.to_undirected()

# 使用 Louvain 算法进行社团划分
partition = community_louvain.best_partition(G_undirected)

# 计算模块度
modularity = community_louvain.modularity(partition, G_undirected)
print("Modularity:", modularity)
# 输出社区数量
num_communities = len(set(partition.values()))
print("Number of communities:", num_communities)

# 输出每个社区的节点编号
for community_id in set(partition.values()):
    nodes_in_community = [node for node, comm_id in partition.items() if comm_id == community_id]
    print(f"Community {community_id}: {nodes_in_community}")

# 计算边的权重（使用无向图 G_undirected）
edge_weights = {(u, v): sum(data['weight'] for _, _, data in G.edges(u, data=True))
                for u, v in G_undirected.edges()}

# 创建节点大小列表（与 PageRank 成正比）
node_sizes = [pagerank[node] * 1200 for node in G.nodes()]

# 创建边粗细列表（与边的权重成正比）
edge_widths = [edge_weights[(u, v)] * 0.1 for u, v in G_undirected.edges()]

pos = nx.spring_layout(G_undirected, k=0.15)
#pos = nx.kamada_kawai_layout(G)



# 设置节点颜色，确保每个社区的节点赋予相同的颜色
node_colors = [partition[node] for node in G_undirected.nodes()]

#获取社区列表
communities = set(partition.values())
#为每个社区分配一个唯一的颜色
color_map = plt.cm.get_cmap('tab10', len(communities))

# 获取社团划分
partition = community_louvain.best_partition(G_undirected)

# 根据社团数量创建颜色列表
num_communities = len(set(partition.values()))
color_map = plt.cm.get_cmap('tab20', num_communities)  # 使用 'tab20' cmap 并根据社团数量调整

# 为每个节点分配颜色
node_colors = [color_map(partition[node]) for node in G_undirected.nodes()]
# 绘制节点
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=color_map)

# 为每条边创建一个颜色列表，颜色与其连接的节点社团相同
edge_colors = []
for u, v in G_undirected.edges():
    u_community = partition[u]
    v_community = partition[v]
    # 选择两个社团中的颜色，这里我们选择 u 的社团颜色
    edge_color = color_map(u_community)
    edge_colors.append(edge_color)

# 绘制边
#nx.draw_networkx_edges(G, pos, edgelist=G_undirected.edges(), width=edge_widths, alpha=0.5, edge_color=edge_colors)
# 绘制边，使用普通箭头样式
nx.draw_networkx_edges(G, pos, edgelist=G_undirected.edges(), width=edge_widths, alpha=1, edge_color=edge_color)
# 绘制节点标签
nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

# 初始化一个变量来存储度的和
degree_sum = 0

# 遍历图中的所有节点
for node in G_undirected.nodes():
    # 获取节点的度
    degree = G_undirected.degree(node)
    # 计算度的和
    degree_sum += degree * (degree - 1)

# 输出所有节点的度的和
print("The sum of all degrees:", degree_sum)

# 关闭坐标轴
plt.axis('off')
plt.show()


