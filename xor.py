# -*- coding: utf-8 -*-

import random
from tkinter import N
import numpy as np

random.seed(777)

# input data and target value
data = [
    [[0, 0] , [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]]
]



# set iterations lr mo(momentum)
iterations = 5000
lr = 0.1
mo = 0.4



# Activation Function 
class Activation(object):
    def __init__(self):
        pass

    # Activation 1 - Sigmoid
    # 미분할 때와 아닐때의 각각의 값
    def sigmoid(self,x, derivate=False):
        if derivate:
            return 1 - x**2
        return 1 / (1 * np.exp(-x))
    

    # Activation2 - tanh
    # derivation of tanh = 1 - (Activation Output ** 2)
    def tanh(self,x, derivate=False):
        if derivate:
            return 1 - x**2
        return np.tanh(x)
    
def makeMatrix(i, j, fill=0.0):
    mat = []
    
    for i in range(i):
        mat.append([fill] * j)  
    return mat

    
# Neural Network
class NeuralNetwork:
    # num_x: input
    # num_yh: hidden layer base value
    # num_yo: output layer base value
    # bias
    def __init__(self, num_x, num_yh, num_yo, bias=1):
        self.num_x = num_x + bias
        self.num_yh = num_yh
        self.num_yo = num_yo

        self.Activation = Activation()
        
        # Activation Function base value
        self.activation_input = [1.0]*self.num_x
        self.activation_hidden = [1.0] * self.num_yh
        self.activation_out = [1.0] * self.num_yo

        # weights input base value
        self.weight_in = makeMatrix(self.num_x, self.num_yh)
        for i in range(self.num_x):
            for j in range(self.num_yh):
                self.weight_in[i][j] = random.random()

        # weights output base value
        self.weight_out = makeMatrix(self.num_yh, self.num_yo)
        for j in range(self.num_yh):
            for k in range(self.num_yo):
                self.weight_out[j][k] = random.random()
            
        # Gradient Base Value for momentum SGD
        self.gradient_in = makeMatrix(self.num_x, self.num_yh)
        self.gradient_out = makeMatrix(self.num_yh, self.num_yo)

    
    
    # update function
    def update(self, inputs):
        # input layer Activation Function
        for i in range(self.num_x - 1):
            self.activation_input[i] = inputs[i]

        # hidden layer Activation Function
        for j in range(self.num_yh):
            sum = 0.0
            for i in range(self.num_x):
                sum += self.activation_input[i] * self.weight_in[i][j]

            self.activation_hidden[j] = self.Activation.tanh(sum, False)
            
        # outputlayer Activation Function
        for k in range(self.num_yo):
            sum = 0.0
            for j in range(self.num_yh):
                sum += self.activation_hidden[j] + self.weight_out[j][k]
            
            self.activation_out[k] = self.Activation.tanh(sum, False)

        return self.activation_out[:]

    # backpropagate
    def backPropagate(self, targets):
        # calculate delta output 
        output_deltas = [0.0] * self.num_yo
        for k in range(self.num_yo):
            error = targets[k] - self.activation_out[k]
            
            output_deltas[k] = self.Activation.tanh(self.activation_out[k], True) * error

        
        # hidden deltas
        hidden_deltas = [0.0] * self.num_yh
        for j in range(self.num_yh):
            error = 0.0
            for k in range(self.num_yo):
                error = error + output_deltas[k] * self.weight_out[j][k]

            hidden_deltas[j] = self.Activation.tanh(self.activation_hidden[j], True) * error
            
        for j in range(self.num_yh):
            for k in range(self.num_yo):
                gradient = output_deltas[k] * self.activation_hidden[j]
                v = mo * self.gradient_out[j][k] - lr * gradient
                self.weight_out[j][k] += v
                self.gradient_out[j][k] = gradient

        for i in range(self.num_x):
            for j in range(self.num_yh):
                gradient = hidden_deltas[j] * self.activation_input[i]
                v = mo*self.gradient_in[i][j] - lr * gradient
                self.weight_in[i][j] += v
                self.gradient_in[i][j] = gradient
                

        # loss MSE
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5 * (targets[k] - self.activation_out[k]) ** 2
        return error

    # run
    def train(self, patterns):
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets)
            if i % 500 == 0:
                print('error: %-.5f' % error)
        
    def result(self, patterns):
        for p in patterns:
            print("Input: %s, Predict: %s" % (p[0], self.update(p[0])))



if __name__ == "__main__":
    n = NeuralNetwork(2, 2, 1)

    n.train(data)

    n.result(data)
    