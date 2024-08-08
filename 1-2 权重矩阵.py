import numpy as np


# relu激活函数
def activate_relu(inputs):
    return np.maximum(0, inputs)


# 生成randn正态分布的权重矩阵
def create_weights(n_inputs, n_neurons):
    return np.random.randn(n_inputs, n_neurons)


# 随机生成偏置值
def create_bias(n_neurons):
    return np.random.randn(n_neurons)


# 输入矩阵
a11 = -1.4
a12 = 1.5
a13 = 1.1

a21 = -0.2
a22 = 0.5
a23 = 0.4

a31 = 0.1
a32 = -0.6
a33 = 0.5

inputs = np.array([[a11, a12, a13],
                  [a21, a22, a23],
                  [a31, a32, a33]])

# 偏置
b1 = create_bias(2)

# 权重矩阵
weights = create_weights(3, 2)

# dot运算
sum1 = np.dot(inputs, weights) + b1

print(activate_relu(sum1))
