import numpy as np


# 定义激活函数
def activate_relu(inputs):
    return np.maximum(0, inputs)


# 定义层类
class layer:
    def __init__(self, n_input, n_neurons):
        self.weights = np.random.randn(n_input, n_neurons)
        self.bias = np.random.randn(n_neurons)

    def layer_forward(self, inputs):
        sum1 = np.dot(inputs, self.weights) + self.bias
        output = activate_relu(sum1)
        return output

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
layer1 = layer(2, 3)

# 第二层
layer2 = layer(3, 4)

# 第三层
layer3 = layer(4, 2)

# 第一层运算
output1 = layer1.layer_forward(inputs)

# 第二层运算
output2 = layer2.layer_forward(output1)

# 第三层运算
output3 = layer3.layer_forward(output2)
print("输出为：", output3)
