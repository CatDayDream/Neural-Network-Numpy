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


# 输入层
a11 = 0.4
a12 = 0.2

a21 = 1.5
a22 = 1.9

a31 = 2.1
a32 = 0.4

a41 = 0.6
a42 = -2.3

a51 = -0.2
a52 = 0.9

inputs = np.array([[a11, a12],
                   [a21, a22],
                  [a31, a32],
                  [a41, a42],
                  [a51, a52]])
print("输入为:", inputs)

# 第一层
weights1 = create_weights(2, 3)
bias1 = create_bias(3)

# 第二层
weights2 = create_weights(3, 4)
bias2 = create_bias(4)

# 第三层
weights3 = create_weights(4, 2)
bias3 = create_bias(2)

# 第一层运算
sum1 = np.dot(inputs, weights1) + bias1
output1 = activate_relu(sum1)
print("第一层输出为：", output1)

# 第二层运算
sum2 = np.dot(output1, weights2) + bias2
output2 = activate_relu(sum2)
print("第二层输出为：", output2)

# 第三层运算
sum3 = np.dot(output2, weights3) + bias3
output3 = activate_relu(sum3)
print("第三层输出为：", output3)
