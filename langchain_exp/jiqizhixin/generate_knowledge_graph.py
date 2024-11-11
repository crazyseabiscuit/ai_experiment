from pyvis.network import Network
import networkx as nx
import matplotlib.pyplot as plt
import json
from datetime import datetime, timezone, timedelta
import argparse


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Generate knowledge graph with date filtering"
    )
    parser.add_argument(
        "--start-date", type=str, help="Start date in YYYY-MM-DD format", default=None
    )
    parser.add_argument(
        "--end-date", type=str, help="End date in YYYY-MM-DD format", default=None
    )
    return parser.parse_args()


def parse_date(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S %z")


def filter_by_date(articles, start_date=None, end_date=None):
    if not start_date and not end_date:
        return articles

    filtered_data = []
    for article in articles:
        article_date = parse_date(article["date"])

        if start_date and end_date:
            if start_date <= article_date <= end_date:
                filtered_data.append(article)
        elif start_date:
            if article_date >= start_date:
                filtered_data.append(article)
        elif end_date:
            if article_date <= end_date:
                filtered_data.append(article)

    return filtered_data


plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",
    "SimHei",
    "DejaVu Sans",
]

plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

INPUT_FILE = "jiqizhixin_articles.json"
data = []
with open(INPUT_FILE, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip().rstrip(",")
        if line and line not in ["[", "]"]:
            try:
                json_obj = json.loads(line)
                if isinstance(json_obj, list):
                    data.extend(json_obj)
                else:
                    data.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")


print(f"Total items loaded: {len(data)}")


def get_date_with_tz(date_str):
    if not date_str:
        return None
    tz = timezone(timedelta(hours=8))  # 设置为东八区
    date = datetime.strptime(date_str, "%Y-%m-%d")
    # if "end" in date_str:  # 如果是结束日期，设置为当天最后一秒
    #     date = date.replace(hour=23, minute=59, second=59)
    return date.replace(tzinfo=tz)


args = parse_cli_args() 
start_date = get_date_with_tz(args.start_date)
end_date = get_date_with_tz(args.end_date)
print(f"Start date: {start_date}, End date: {end_date}")

filtered_data = filter_by_date(data, start_date, end_date)
print(f"Articles after date filtering: {len(filtered_data)}")

data = filtered_data
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
