import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):             # layer initialization
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons)  #rows(input count) x columns(neuron count) -  0.01 to ensure minimal weight initialization
        self.biases = np.zeros((1,n_neurons))            # 2 parameters for np.zeros row ( 1 row) x column (number of neurons)
 
    def forward(self, inputs):        # initialize forward pass 
        self.output = np.dot(inputs,self.weights) + self.biases     # Matrix Mult. of inputs against weights added with broadcasted biases

class Activation_ReLU:       # ReLU to introduce non-linearity inbetween multiplication layers
    def forward(self,inputs):   
        self.output = np.maximum(0,inputs)   # same dimension as input matrix, replaces value with 0 if < 0

class Activation_Softmax:     #SoftMax activation used for probability final activation layer (due to spiral classification problem )
    def forward(self,inputs):     
        # e^x conversion of inputs in the matrix, all subtracted by the largest value in order to taper magnitude 
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))     # axis = 1 returns a matrix #neurons x 1 of the max value per row 
        # divide the e^x conversion by the summation of the row (each element in the row is a different neuron output) 
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims = True)   #axis = 1 to maintain #neurons x 1 dimensions 
        self.output = probabilities

class Activation_Sigmoid:
    def forward(self,inputs):
        self.output = 1/(1+np.exp(-inputs))

class Activation_Tanh:
    def forward(self,inputs):
        self.output = (2/(1+np.exp(-2 * inputs)) - 1)

class Loss:
    def calculate(self, output, y_values):
        sample_losses = self.forward(output,y_values)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true,axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


X,y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,3)

activation1 = Activation_ReLU()

dense2 = Layer_Dense(3,3)

activation2 = Activation_Softmax()

dense3 = Layer_Dense(3,3)

loss_function = Loss_CategoricalCrossEntropy()
#singular forward pass against 2 layers with one ReLU and a concluding softmax 
dense1.forward(X)
activation1.forward(dense1.output)
dense3.forward(activation1.output)
activation1.forward(dense3.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

#Non propogated biases and weights for this run 
print(activation2.output[:5])

loss = loss_function.calculate(activation2.output, y)
print("loss: ", loss)

predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) ==2:
    y = np.argmax(y,axis=1)
accuracy = np.mean(predictions == y)
print("acc:", accuracy)

# plt.scatter(X[:,0], X[:,1], c=y, cmap='brg')
# plt.show()

# A = [[1,2,3],[4,155,6],[7,8,9]]
# print(  np.max(A,axis=1))
   
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
