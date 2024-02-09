transformer and bert is contributed by google 
gpt,gpt2 and gpt3 are contributed by openai
open ai 是想解决强ai技术问题，也就是说想解决一个更大的问题，transformer是想解决一个机器翻译的问题，gpt stands for generative pre-training
 
## gpt paper 概要介绍，作者是怎么解决这个问题的呢？
gpt是使用有监督的微调 将transformer的最后一层拿出来然后乘以一个参数来完成
对于下游任务
# Classification 
# Entailment(蕴含)
# Similarity
# Multiple Choice

## experiment
使用bookscorpus数据集上被训练出来的，这上面有7000本没有被发表的书
使用了12层的解码器，每一层的维度是768.
bert的解码器也是12层，每一层的维度是768.bert base就是为了和gpt做对比,bert large是bert base参数的三倍主要是因为它使用的数据集的大小比gpt的数据集大三倍，

## GPT2 paper language model are unsupervised Multitask Learner
使用webtext数据集，主要卖点是zero shot
commoncrawler是一个抓取公开页面的工具


## gpt-3 language are few shot learners
gpt3 不需要做微调和计算梯度

# Multitask learmnimg amd reinforcement learning for personalized dialog generation
# a empircal study
the evaluation metric of dialog systems are not used when training the seq2seq model 
