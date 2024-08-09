import numpy as np

NETWORK_SHAPE = [2, 3, 4, 2]


# 定义激活函数
def activate_relu(inputs):
    return np.maximum(0, inputs)


# 定义层类
class Layer:
    def __init__(self, n_input, n_neurons):
        self.weights = np.random.randn(n_input, n_neurons)
        self.bias = np.random.randn(n_neurons)

    def layer_forward(self, inputs):
        sum1 = np.dot(inputs, self.weights) + self.bias
        output = activate_relu(sum1)
        return output


# 定义网络类
class Network:
    def __init__(self, network_shape):
        self.layers = []
        for i in range(len(network_shape) - 1):
            layer = Layer(network_shape[i], network_shape[i+1])
            self.layers.append(layer)

    # 定义前馈运算函数
    def network_forward(self, inputs):
        outputs = [inputs]
        for i in range(len(self.layers)):
            layer_output = self.layers[i].layer_forward(outputs[i])  # 迭代
            outputs.append(layer_output)
        return outputs


if __name__ == "__main__":
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

    # batch
    inputs = np.array([[a11, a12],
                       [a21, a22],
                       [a31, a32],
                       [a41, a42],
                       [a51, a52]])
    print("输入为：\n", inputs)
    network = Network(NETWORK_SHAPE)
    outputs = network.network_forward(inputs)
    print("第一层输出：\n", outputs[1])
    print("第一层输出：\n", outputs[2])
    print("最终输出为：\n", outputs[3])
