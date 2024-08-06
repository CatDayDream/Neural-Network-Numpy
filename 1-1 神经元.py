# Python手搓神经网络
# 1-1 神经元
import numpy as np

# 输入层
a1 = 0.9
a2 = 0.5
a3 = 0.6
inputs = np.array([a1, a2, a3])

# 权重矩阵
w1 = 0.4
w2 = 0.7
w3 = -0.4
weights = np.array([[w1],
                   [w2],
                   [w3]])

# 偏置
b1 = 0.5

# dot运算
sum1 = np.dot(inputs, weights) + b1


# 激活函数
def activate_relu(inputs):
    return np.maximum(0, inputs)


print(activate_relu(sum1))
