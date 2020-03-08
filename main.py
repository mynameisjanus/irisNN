import numpy as np
# import irisneuralnet as IRN
import nnpytorch
import torch
import torch.nn as nn

# Download the dataset
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# Run kNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# Run Handcrafted Neural Network
# x = IRN.NeuralNet(X_train, y_train, hidden_layer = 200, lr = 0.001, epochs = 100)
# x.train_neural_network()
# nn_score = x.test_neural_network(X_test, y_test)

# Run Neural Network with PyTorch

train_data = {'x': torch.tensor(X_train, dtype = torch.float32), 'y': torch.tensor(y_train, dtype = torch.long)}
test_data = {'x': torch.tensor(X_test, dtype = torch.float32), 'y': torch.tensor(y_test, dtype = torch.long)}


input_size = 4
hidden_layer = 200
output_size = 3
n_epochs = 100

model = nn.Sequential(
        nn.Linear(input_size, hidden_layer),
        nn.ReLU(),
        nn.Linear(hidden_layer, output_size),
        nn.LogSoftmax(dim=1),
        )

nnpytorch.train_model(train_data, model)
loss, accuracy = nnpytorch.run_epoch(test_data, model.eval(), None)
print(accuracy)

# Compare errors
print("kNN test score: {:.4f}".format(np.mean(y_pred == y_test)))
# print("Neural Network test score: {:.4f}".format(nn_score))