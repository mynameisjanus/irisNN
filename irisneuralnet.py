# Neural Network

import numpy as np

def relu(x):
        return np.maximum(0, x)

def drelu(x):
        return 1 * (x > 0)

def output_layer_activation(x):
        return x

def output_layer_activation_derivative(x):
        return 1
        
class NeuralNet():
        
        def __init__(self, X_train, y_train, hidden_layer, lr, epochs):
                self.W1 = np.random.randint(10, size = (hidden_layer, X_train.shape[1]))
                self.W2 = np.ones(hidden_layer)
                self.b = np.zeros(hidden_layer)
                self.learning_rate = lr
                self.epochs = epochs
                self.training_points = X_train
                self.ypoints = y_train
                
        def train(self, X, y):
                f1 = np.vectorize(relu)
                df1 = np.vectorize(drelu)
                
                # Forward propagation
                z1 = self.W1 @ X + self.b
                a1 = f1(z1)
                z2 = self.W2 @ a1
                output = output_layer_activation(z2)
                
                # Backpropagation
                output_layer_error = (output - y) * output_layer_activation_derivative(z2)
                hidden_layer_error = np.multiply(self.W2 * output_layer_error, df1(z1))
                
                # Gradients
                b_grad = hidden_layer_error
                W2_grad = (a1 * output_layer_error).T
                W1_grad = np.outer(hidden_layer_error, X)
                
                # Update the parameters
                self.b = self.b - self.learning_rate * b_grad
                self.W1 = self.W1 - self.learning_rate * W1_grad
                self.W2 = self.W2 - self.learning_rate * W2_grad
                
        def predict(self, X):
                f1 = np.vectorize(relu)
                
                z1 = self.W1 @ X + self.b
                a1 = f1(z1)
                z2 = self.W2 @ a1
                activated_output = output_layer_activation(z2)
                
                return activated_output.item()
        
        def train_neural_network(self):
                for epoch in range(self.epochs):
                        for x, y in zip(self.training_points, self.ypoints):
                                self.train(x, y)
        
        def test_neural_network(self, X_test, y_test):
                y_pred = []
                for point in X_test:
                        y_pred.append(self.predict(point))
                return np.mean(y_pred == y_test)