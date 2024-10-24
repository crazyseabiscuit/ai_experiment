import json
from pyvis.network import Network
import networkx as nx
import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties

# 设置中文字体
plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",
    "SimHei",
    "DejaVu Sans",
]  # 优先使用这些字体

plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号
input_file = 'jiqizhixin_articles.json'
with open(input_file, 'r', encoding='utf-8') as file:
    data = json.load(file)


# 创建一个 Pyvis 网络
net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")

# 添加节点
for item in data:
    net.add_node(
        item["title"],
        label=item["title"],
        title=f"Author: {item['author']}\nCategory: {item['category']}",
        color="#97C2FC",
    )

# 添加边
for i, item_i in enumerate(data):
    for j, item_j in enumerate(data[i + 1 :], start=i + 1):
        if item_i["author"] == item_j["author"]:
            net.add_edge(
                item_i["title"], item_j["title"], title="Same Author", color="#FFFF00"
            )
        if item_i["category"] == item_j["category"]:
            net.add_edge(
                item_i["title"], item_j["title"], title="Same Category", color="#00FFFF"
            )

# 配置物理布局
net.force_atlas_2based()

# 添加交互控制选项
net.show_buttons(filter_=["physics"])

# 保存为 HTML 文件
net.save_graph("interactive_knowledge_graph.html")


# 创建一个空的无向图
G = nx.Graph()

# 添加节点
for item in data:
    G.add_node(item["title"], author=item["author"], category=item["category"])

# 添加边，连接相同作者的文章
for i, item_i in enumerate(data):
    for j, item_j in enumerate(data[i + 1 :], start=i + 1):
        if item_i["author"] == item_j["author"]:
            G.add_edge(item_i["title"], item_j["title"], label="same_author")
        if item_i["category"] == item_j["category"]:
            G.add_edge(item_i["title"], item_j["title"], label="same_category")

# 绘制图形
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=2, iterations=50)
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color="lightblue",
    font_size=8,
    node_size=1000,
    edge_color="gray",
)
labels = nx.get_edge_attributes(G, "label")
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

# 保存为图片
plt.savefig("knowledge_graph.png", dpi=300, bbox_inches="tight")

# 显示图形
plt.show()
