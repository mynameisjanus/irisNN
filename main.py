import numpy as np
import irisneuralnet as IRN

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

# Run Neural Network
x = IRN.NeuralNet(X_train, y_train, hidden_layer = 100, lr = 0.001, epochs = 300)
x.train_neural_network()
nn_score = x.test_neural_network(X_test, y_test)

# Compare errors
print("kNN test score: {:.4f}".format(np.mean(y_pred == y_test)))
print("Neural Network test score: {:.4f}".format(nn_score))