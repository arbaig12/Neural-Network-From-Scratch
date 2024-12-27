import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):             # layer initialization
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))            # 2 parameters for np.zeros row x column
 
    def forward(self, inputs):        # initialize forward pass 
        self.output = np.dot(inputs,self.weights) + self.biases   

X,y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,3)
dense1.forward(X)

print(dense1.output[:5])
 
# biases = [2,3,0.5]
# weights = [[0.2,0.8,-.5,1], [0.5,-0.91,0.26,-0.5], [-0.26,-0.27,0.17,0.87]]
# inputs = [[1,2,3,2.5],[2,5,-1,2],[-1.5,2.7,3.3,-.8]]

# biases2 = [-1,2,-0.5]
# weights2 = [[0.1,-0.14,0.5],[-0.5,0.12,-0.33],[-0.44,0.73,-0.13]]


# # for bias, weight in zip(biases, weights):
# #     output = bias  
# #     for input_value, weight_value in zip(inputs, weight):
# #         output += input_value * weight_value  
# #     print(output)

# output1 = np.dot(inputs,np.array(weights).T) + biases
# output2 = np.dot(output1,np.array(weights2).T) + biases2
# print(output2)
