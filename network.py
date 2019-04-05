import numpy as np
import pandas as pd
%matplotlib inline

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(1-sigmoid(x))

def loss(x,y):
    return (x-y)**2



class NeuralNet():
    def __init__(self, input = 1, neurons = 10, weights = False):
        self.input = 1
        self.neurons = 10
        if not weights:
            print("Random weights")
            self.weights = [np.random.random(size = neurons)*2,
                            np.random.random(size = input)*2]
            print(self.weights)
        else:
            self.weights = weights
        self.activation = sigmoid
        self.loss = loss
        self.loss_hist = []

    def Forward(self, data):
        # print(self.weights[0])
        # print("data:", data)
        # print("weights:", self.weights[0], self.weights[0].shape)
        result = np.dot(self.weights[1], data)
        return result

    def fit(self, X_train, y_train):
        z = self.Forward(X_train)
        # print("printing", z)
        a = self.activation(z)
        loss = self.loss(a, y_train)
        # print(f"Loss:{sum(loss)}")
        grad = 2*(a - y_train)*sigmoid_der(z)*a
        self.weights -= grad * 0.01
        self.loss_hist.append(np.sum(loss))

    def updateWeights(self):
        pass

# %% You know nothing

nn = NeuralNet(neurons = 50)



#%%

for i in range(100000):
    X_train = np.random.random()*20
    y_train = np.sin(X_train)
    nn.fit(X_train, y_train)

print(nn.weights)

pd.Series(nn.loss_hist).ewm(alpha = 0.01).mean().plot()


#%%

X_test = np.linspace(0,2*np.pi, 100)
y_test = np.sin(X_test)

np.dot(nn.weights)
