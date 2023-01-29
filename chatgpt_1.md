## 数据来源
1. 文本数据，自己做人工标注。
2. 人工标注的，说明，然后有相应的问题和回答的pair
3. 使用open ai的用户提供的user case，然后进行人工标注，对于每个用户最多采用200个问题，一个用户可能会问一些相似的问题，所以根据用户来划分是一种比较好的方式，PII的数据也是会被过滤掉。


GPT是在最后的一层加个softmax来输出一个概率。chatgpt是在后面加一个线性层来投影，是一个输出为1的线性层

reward modeling：使用大模型训练的时候，不稳定的问题依然没有解决，loss function会炸掉

关于k的取值k=4还是k=9


##TODO 
context based Q&A and Reinforcement learning demo


？Rsita