import numpy as np


# 定义softmax激活函数
def activate_softmax(inputs):
    max_values = np.max(inputs, axis=1, keepdims=True)  # axis=1取每行最大值
    slided_inputs = inputs - max_values
    return slided_inputs


class Layer:
    def __init__(self, n_input, n_neurons):
        self.weights = np.random.randn(n_input, n_neurons)
        self.bias = np.random.randn(n_neurons)

    def layer_forward(self, inputs):
        sum1 = np.dot(inputs, self.weights) + self.bias
        output = activate_softmax(sum1)
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
