# AI 2022 the darkness before dawn

## table of contents
- AI Content Generation 
- Faiththful with Graph methdology 
- Computer Vision
- AI for Science
- HardWare and Engineering 
- Existing Bottle neck
- AI future trending and 2023 prediction

AI solution is integrated with cloud services and open source framwork/software now, engineering is easy to use them now.we can just download a pretrained model, use just a small dataset to train a machine learning model, even more we can use a existing software or cloud service to fullfill the AI feature for the requirments, so we don't need to train a model by ourself at all. Not only normal size company, in the early 2022,but also black industry they use deepface to replace the images and videos human faces to scam and fraud in domestic. It seems AI technology is very promising and completed for all of us,why the title is "AI 2022 the darkness before dawn"? Why industry agree there are still many blockers from using these technololy to create bussiness values?
The short answer is there are still many limitations, AI industry is developing very fast in this year, some domains such as content genration is very popular, multimodal large scale model is definitly one of the AI hostspot, many person are talking about, it also has good performance such as stableDiffusion, chatGPT, in China there are also many institution, company realse large scaled models, there performance are good.We will go through  
AI content generation model like chatGPT.We need to understand the basic theory is still base on probability, for example when we want to know whether it will rain or not tomorrow, so we observe the history data and find when umbrella and the ground is wet it always rain, is this always right? AI content generation has a probability to generate incorrect content. Industry is still figuring out how to fully solve this problem,it will take a while.
### AI Content Generation 
Chatgpt: Good points are it achive good dialogue performance especially in dialogue introduction fields, it collect demonstration data, and train a supervised policy optimizing language models for dialogues, this improve truthfulness over GPT-3 and has small improvements in toxicity but not bias. it demonstrate how to fullly or better use the labled data for chat.
        drawbacks are also still existing, first of the new methodology used in ChapGPT can improve the truethfulness and reduce toxicity, but i can't totally avoid, the new method methdology is still based probility methdology
        it still make simple mistake
        Personalize is a important feature for AI application, but for ChatGPT, personalize is not possible for now. I think in the future maybe we can computer all the intense calculations in server side, the personalize can happen in edge device. 
AIGC progress, if you remember about two years ago, AI can do some computer vision task such as faceswap for video and image(see faceswapzai in referernce application section),For now, machine can do text to pictures and text to video, 3D generation etc.
stable diffusion: Stable Diffusion is a deep learning, text-to-image model released in 2022. It is primarily used to generate detailed images conditioned on text descriptions, though it can also be applied to other tasks such as inpainting, outpainting, and generating image-to-image translations guided by a text prompt.

## Faiththful with Graph methdology 
Accoding to my observation, Graph transformer is one of the most important achivement in GNNs, 
diffusion model, how to merge it in Gragh,fake detection and EDA related task.indusrty is focusing not traditional graph problem, how to sovle it with grah methdology and GNN
Graph Transformer:
        Graph Neural Networks (GNNs) have been widely applied to various fields due to their powerful representations of graph-structured data. Despite the success of GNNs, most existing GNNs are designed to learn node representations on the fixed and homogeneous graphs. The limitations especially become problematic when learning representations on a misspecified graph or a heterogeneous graph that consists of various types of nodes and edges. To address this limitations, propose Graph Transformer Networks (GTNs) that are capable of generating new graph structures, which preclude noisy connections and include useful connections (e.g., meta-paths) for tasks, while learning effective node representations on the new graphs in an end-to-end fashion. We further propose enhanced version of GTNs, Fast Graph Transformer Networks (FastGTNs), that improve scalability of graph transformations. Compared to GTNs, FastGTNs are 230x faster and use 100x less memory while allowing the identical graph transformations as GTNs. In addition, we extend graph transformations to the semantic proximity of nodes allowing non-local operations beyond meta-paths. Extensive experiments on both homogeneous graphs and heterogeneous graphs show that GTNs and FastGTNs with non-local operations achieve the state-of-the-art performance for node classification tasks.
## Computer Vision
Vision transformer (ViT)
        While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.

## AI for Science
The most exited news of AI for science in 2023 is deepmind protein remodeling  AI + Biology natureL: 
The entire protein universe’: AI predicts shape of nearly every known protein. DeepMind’s AlphaFold tool has determined the structures of around 200 million proteins , detail information can be found in this link https://www.nature.com/articles/d41586-022-02083-2
## HardWare and Engineering 
AutoML： AWS gluon，让机器去寻找最好的神经网络，多目标，而且模型能够轻量。MLOps更加精细化，从数据开始管理。
AutoML: AWS gluon, let the machine find the best neural network, multi-objective, and the model can be lightweight. MLOps is more refined, starting from data management.

From an engineering point of view, how to make algorithmic personnel deal with hardware as little as possible, how to combine heterogeneous network CPU and GPU structures, and use the capabilities of the system to do. google gsharp and Berkeley alpha implement feature like model cutting , make full use of resources. Make all tasks fill more full occupied.
Hardware and asynchronization, the combination of specific chips and high-level networks, using system capabilities, model as service, can schools cooperate with enterprises. Schools have more room for imagination

## Existing Bottle neck
We are still in narrow AI period era, strong reliance on data,high computing cost is still high
the AI develop is not very balanced, huge companies they have enough computing, data and human resource, but other company of institutions doesn't have enough
most ML models are still memory model, it doesn't perform very well on unseen data, how to explain, how to be small and ROI efficient

the nature of AI does not change, during the testing some of the AI response does not make sense, some tech can reduce this kind of proflem but we can't resolve the problem fundamentally
We should know the boundary of AI, it is important for us to make the final decision.

可解释，可信，可控可靠
前后逻辑和事实不符合，范式的改变去解决这些问题，这些困难很难克服，把能用的地方先用起来
可以有大规模图数据的场景是比较少的，能做大规模场景下的图数据的实验少，从学术界的角度也很尴尬，高校基本上是被排除在之外的，更多的 人工智能的范式如果是百花齐放的话，应该会更好一些。
Explainable, credible, controllable and reliable
The front and rear logics do not conform to the facts, and the paradigm changes to solve these problems. These difficulties are difficult to overcome, so we should use the places that can be used first.
There are relatively few scenes that can have large-scale graph data, and there are few experiments that can do large-scale graph data in large-scale scenarios. It should be better if the paradigm of intelligence is full of flowers.

从工程的角度上来说，怎么样让算法人员尽量的少的去和硬件打交道，怎么样把异构的网络cpu和GPU结构组合好，利用系统的能力去做google gsharp 伯克利alpha，做模型的切割，充分的用资源。让所有的任务fill的比较饱满
硬件话和异步化，特定的芯片和高层次的网络结合，用系统的能力，model as service，学校是不是可以跟企业做合作。学校的想象空间更大
From an engineering point of view, how to make algorithmic personnel deal with hardware as little as possible, how to combine heterogeneous network cpu and GPU structures, and use the capabilities of the system to do google gsharp Berkeley alpha and model cutting , make full use of resources. Make all tasks fill more full
Hardware and asynchronization, the combination of specific chips and high-level networks, using system capabilities, model as service, can schools cooperate with enterprises. Schools have more room for imagination

With a credible graph neural network, the large model GPT-4 will have some more amazing performances in the direction of content generation. The security of the graph neural network is also an urgent problem to be solved by artificial intelligence in the future.
Will there be better results by integrating these large models? How does the large model make some contribution to the small and precise model
The versatility of AI directly reflects a kind of ability, which is different from the previous generalization
It's all about whether the large model can focus on one field, and the effect of doing an integration will be better
International geopolitics, technology open source issues such as chips
## AI Industry Trending
可信的图神经网络，大模型gpt4会有一些更加惊艳的表现，在内容产生的方向。图神经网络的安全性，也是未来一段时间人工智能急切要解决的问题
把这些大模型做集成会不会有更好得结果？大模型怎么样对小而且精得模型做出一些贡献
AI得通用性，直接就体现出一种能力，和之前得泛化性是不一样得
都是大模型可否专注在一个领域上，这样做一个集成得效果还会好一点
国际地缘政治，芯片等技术开源得问题

With a credible graph neural network, the large model gpt4 will have some more amazing performances in the direction of content generation. The security of the graph neural network is also an urgent problem to be solved by artificial intelligence in the future.ni 
Will there be better results by integrating these large models? How does the large model make some contribution to the small and precise model
The versatility of AI directly reflects a kind of ability, which is different from the previous generalization
It's all about whether the large model can focus on one field, and the effect of doing an integration will be better
International geopolitics, technology open source issues such as chips

## reference
- Vision Transformer https://arxiv.org/abs/2010.11929
- faceswapzai https://faceswap.dev//
- AlphaFold remodel: https://www.nature.com/articles/d41586-022-02083-2
- ChatGPT https://openai.com/blog/chatgpt/
- MAKE A VIDEO  https://makeavideo.studio/
- CogVideo Large-scale Pretraining for Text-to-Video Generation via Transformers  https://models.aminer.cn/cogvideo/
- TOME generative storytelling text context based on power point  https://beta.tome.app/
- GODEL: finetuning phases that require infomration external to the current conversation(e.g., a database or document) to produce good responses


## Draft 
## Casual Inference:
Related topic : AIGC, big model,privacy computing, Gragh and Machine learning
Industry hot topic: digital transformation
causality 

Recently the industry is focused on if you have a good pretrain model, how to make it best use for subtask for different bussiness scenarios.
Mutimodel trending : leverage different type of data to improve the subtask, Chinese Academy of Sciences use voice and video data to judge the position of person at night or no enough fiber 
AI design relted works such as image generation, video generation
chatGPT and google Lamda almost pass the turing test
KG
faiththful with Graph methdology 
multiple technology are merged together 

AI system can be very commonly used, it can do something directly 
AI for science and AI for industry
AI models growing in complexity and diversity 

make a video和3d生成
### future trending 
nature language explaination 
personalize 
fully understand the business requirements
data simulation is possible? yeah but it is very hard

AI dev ops
hardware accumulation
privacy computing and federation learning 

data preporcessing and business, same scenario you still do them again and again manuelly