# import tensorflow as tf 
import numpy as np 
import os

def sigmoid(z):
    sig = 1/(1+np.exp(-z))
    return sig

def relu(z):
    z = np.array([0,z])
    relu = np.max(z)
    return relu

class Neuron():
    
    def _init__(self, inputs, weights, bias, activation = "sigmoid"):
        self.x = inputs
        self.w = weights
        self.b = bias 
        self.activation = activation

    def output(self, w, x, b, activation):
        if(activation == "sigmoid"):
            y = sigmoid((np.dot(w.T, x)+ b))
            print("Hi")

        elif(activation == "relu"):
            y = relu((np.dot(w, x)+b))
            print("Hello")
        else:
            y = (np.dot(w, x)+ b)     
            print("Hola")
        # print(y)
        return y
    
inputs = np.array([1,2,3,4])
weights = np.array([2,3,4,1])
bias = 3

n1 = Neuron()

print(n1.output(weights, inputs, bias, "relu"))

