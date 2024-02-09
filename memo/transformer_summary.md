

## Model Architechture


### Attention
主要是矩阵乘法采用torch.matmul来进行计算
两种常用的attention的实现

additive attention 使用feed forward network来实现
使用单层的隐藏层来实现，但是使用点积的方法效率更高所以在实际的使用的过程当中都是使用点积来实现

### MultiHeadAttention
多头注意力机制可以允许模型在不同的位置学习到不同的表示信息

## Training


## A First Example


## A Real World Example
