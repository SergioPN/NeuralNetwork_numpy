import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(1-sigmoid(x))

def loss(x,y):
    return (x-y)**2


class NeuralNet():
    def __init__(self, input = 1, neurons = 10):
        self.input = 1
        self.neurons = 10
        self.weights = [np.zeros(neurons)]
        self.activation = sigmoid
        self.loss = loss
        self.loss_hist = []

    def Forward(self, data):
        print(self.weights[0])
        result = np.dot(self.weights[0], data)

    def fit(self, X_train, y_train):
        z = self.Forward(X_train)
        print(z)
        a = self.activation(z)
        loss = self.loss(a, y_train)
        grad = 2*(act - y_train)*sigmoid_der(z)*a
        self.weights -= grad


    def updateWeights(self):
        pass


np.dot(nn.weights[0], X_train)

nn = NeuralNet()

X_train = np.random.random()*20
y_train = np.sin(X_train)


nn.fit(X_train, y_train)


weights = np.zeros(neurons)

np.tanh(4 * weights)
