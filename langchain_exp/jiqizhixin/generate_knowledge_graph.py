import json
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']  # 优先使用这些字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# JSON数据
data = [
    {"title": "1.1亿个结构DFT计算，Meta推出OMat24，AI驱动材料发现开源化", "url": "https://www.jiqizhixin.com/articles/2024-10-22-8", "summary": "近日，Meta 公司推出一个名为「Open Materials 2024」（OMat24）的大型开放数据集和配套的预训练模型，旨在彻底改变 AI 驱动的材料发现。", "author": "ScienceAI", "date": "2024-10-22 15:34:11 +0800", "category": "理论"},
    {"title": "人类自身都对不齐，怎么对齐AI？新研究全面审视偏好在AI对齐中的作用", "url": "https://www.jiqizhixin.com/articles/2024-10-22-7", "summary": "让 AI 与人类价值观对齐一直都是 AI 领域的一大重要且热门的研究课题", "author": "机器之心", "date": "2024-10-22 14:49:42 +0800", "category": "产业"},
    {"title": "AIGC时代如何打击图片造假诈骗？合合信息文档篡改检测有妙招", "url": "https://www.jiqizhixin.com/articles/2024-10-22-6", "summary": "合合信息亮相2024中国模式识别与计算机视觉大会，用AI构建图像内容安全防线", "author": "机器之心", "date": "2024-10-22 14:40:56 +0800", "category": "产业"},
    {"title": "大模型是否有推理能力？DeepMind数月前的论文让AI社区吵起来了", "url": "https://www.jiqizhixin.com/articles/2024-10-22-5", "summary": "最近一段时间，随着 OpenAI o1 模型的推出，关于大型语言模型是否拥有推理能力的讨论又多了起来。", "author": "机器之心", "date": "2024-10-22 13:19:51 +0800", "category": "产业"},
    {"title": "骁龙8至尊版登场：CPU牙膏挤爆，AI生成速度创纪录，奥特曼也来助阵", "url": "https://www.jiqizhixin.com/articles/2024-10-22-4", "summary": "开启终端侧生成式 AI 的全新时代。", "author": "机器之心", "date": "2024-10-22 13:16:43 +0800", "category": "产业"},
    {"title": "DeepSeek新作Janus：解耦视觉编码，引领多模态理解与生成统一新范式", "url": "https://www.jiqizhixin.com/articles/2024-10-22-3", "summary": "我们提出了 Janus，一种基于自回归的多模态理解与生成统一模型。", "author": "机器之心", "date": "2024-10-22 13:14:51 +0800", "category": "产业"},
    {"title": "自动化、可复现，基于大语言模型群体智能的多维评估基准Decentralized Arena来了", "url": "https://www.jiqizhixin.com/articles/2024-10-22-2", "summary": "研究者们已经并陆续构建了成千上万的大规模语言模型（LLM），这些模型的各项能力（如推理和生成）也越来越强。", "author": "机器之心", "date": "2024-10-22 13:11:00 +0800", "category": "工程"},
    {"title": "AI大冰嘴替，骂醒「满脑浆糊」的网友", "url": "https://www.jiqizhixin.com/articles/2024-10-22", "summary": "有了AI，人人都能像大冰一样妙语连珠。", "author": "AI好好用", "date": "2024-10-22 10:25:00 +0800", "category": "产业"},
    {"title": "黄仁勋新访谈：OpenAI是这个时代最具影响力公司之一，马斯克19天创造工程奇迹", "url": "https://www.jiqizhixin.com/articles/2024-10-21-9", "summary": "10 月 4 日，播客节目 BG2（Brad Gerstner 和 Clark Tang）邀请到了英伟达 CEO 黄仁勋，他们一起讨论了 AGI、机器学习加速、英伟达的竞争优势、推理与训练的重要性、AI 领域未来的市场动态、AI 对各个行业的影响、工作的未来、AI 提高生产力的潜力、开源与闭源之间的平衡、马斯克的 Memphis 超级集群、X.ai、OpenAI、AI 的安全开发等。", "author": "机器之心", "date": "2024-10-21 15:17:31 +0800", "category": "产业"},
    {"title": "速度提高1000万倍，AI快速准确预测等离子体加热，助力核聚变研究", "url": "https://www.jiqizhixin.com/articles/2024-10-21-8", "summary": "Sánchez-Villar 的团队开发了 NSTX 和 WEST 上的实时核心离子回旋加速器频率范围 (ICRF) 加热模型。该模型基于两种非线性回归算法，即决策树的随机森林集成和多层感知器神经网络。", "author": "ScienceAI", "date": "2024-10-21 15:01:00 +0800", "category": "理论"},
    {"title": "Nature子刊，北大陈语谦团队提出多模态单细胞数据整合和插补的深度学习方法", "url": "https://www.jiqizhixin.com/articles/2024-10-21-7", "summary": "该团队开发了一种新型的多模态整合方法，能够实现多模态单细胞数据的整合与插补，这一成果可以促进多模态单细胞数据的分析。", "author": "ScienceAI", "date": "2024-10-21 15:00:00 +0800", "category": "理论"},
    {"title": "淘宝百亿补贴“全家桶”桶桶五折！天猫双11今晚8点现货开卖", "url": "https://www.jiqizhixin.com/articles/2024-10-21-6", "summary": "“万元券包”能领尽领！", "author": "新闻助手", "date": "2024-10-21 14:36:32 +0800", "category": "产业"}
]


from pyvis.network import Network
import networkx as nx

# 创建一个 Pyvis 网络
net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")

# 添加节点
for item in data:
    net.add_node(item['title'], 
                 label=item['title'], 
                 title=f"Author: {item['author']}\nCategory: {item['category']}",
                 color="#97C2FC")

# 添加边
for i, item_i in enumerate(data):
    for j, item_j in enumerate(data[i+1:], start=i+1):
        if item_i['author'] == item_j['author']:
            net.add_edge(item_i['title'], item_j['title'], title='Same Author', color="#FFFF00")
        if item_i['category'] == item_j['category']:
            net.add_edge(item_i['title'], item_j['title'], title='Same Category', color="#00FFFF")

# 配置物理布局
net.force_atlas_2based()

# 添加交互控制选项
net.show_buttons(filter_=['physics'])

# 保存为 HTML 文件
net.save_graph("interactive_knowledge_graph.html")


# 创建一个空的无向图
G = nx.Graph()

# 添加节点
for item in data:
    G.add_node(item['title'], author=item['author'], category=item['category'])

# 添加边，连接相同作者的文章
for i, item_i in enumerate(data):
    for j, item_j in enumerate(data[i+1:], start=i+1):
        if item_i['author'] == item_j['author']:
            G.add_edge(item_i['title'], item_j['title'], label='same_author')
        if item_i['category'] == item_j['category']:
            G.add_edge(item_i['title'], item_j['title'], label='same_category')

# 绘制图形
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=2, iterations=50)
nx.draw(G, pos, with_labels=True, node_color='lightblue', font_size=8, node_size=1000, edge_color='gray')
labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

# 保存为图片
plt.savefig('knowledge_graph.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()