import numpy as np
import pandas as pd

# Load the Iris dataset manually (No scikit-learn)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv(url, header=None, names=columns)

# Manually encode class labels (No LabelEncoder)
species_to_int = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
iris['species'] = iris['species'].map(species_to_int)

# Convert dataset to NumPy arrays
X = iris.iloc[:, :-1].values  # Features
y = iris.iloc[:, -1].values  # Target labels

# One-hot encoding for output layer (Manual)
num_classes = len(set(y))
y_one_hot = np.zeros((y.shape[0], num_classes))
for i, label in enumerate(y):
    y_one_hot[i, label] = 1

# Manual train-test split (80-20 split)
def train_test_split_manual(X, y, test_size=0.2):
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = train_test_split_manual(X, y_one_hot)

# Define MLP architecture
input_size = X.shape[1]  # 4 features
hidden_size = 16  # Number of neurons in hidden layer
output_size = num_classes  # 3 classes
learning_rate = 0.1
epochs = 3000

# Xavier/Glorot Weight Initialisation
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1 / input_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1 / hidden_size)
b2 = np.zeros((1, output_size))

# Activation Functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Mini-Batch Gradient Descent
batch_size = 16
num_samples = X_train.shape[0]

for epoch in range(epochs):
    indices = np.random.permutation(num_samples)
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    for i in range(0, num_samples, batch_size):
        X_batch = X_train_shuffled[i:i + batch_size]
        y_batch = y_train_shuffled[i:i + batch_size]

        # Forward Propagation
        Z1 = np.dot(X_batch, W1) + b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = softmax(Z2)

        # Compute Loss (Categorical Cross-Entropy)
        loss = -np.mean(y_batch * np.log(A2 + 1e-8))

        # Backpropagation
        dZ2 = A2 - y_batch
        dW2 = np.dot(A1.T, dZ2) / batch_size
        db2 = np.sum(dZ2, axis=0, keepdims=True) / batch_size

        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * relu_derivative(A1)
        dW1 = np.dot(X_batch.T, dZ1) / batch_size
        db1 = np.sum(dZ1, axis=0, keepdims=True) / batch_size

        # Gradient Descent Update
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Testing the Model
Z1_test = np.dot(X_test, W1) + b1
A1_test = relu(Z1_test)
Z2_test = np.dot(A1_test, W2) + b2
A2_test = softmax(Z2_test)
predictions = np.argmax(A2_test, axis=1)
y_true = np.argmax(y_test, axis=1)

# Compute Performance Metrics
accuracy = np.mean(predictions == y_true)

# Confusion Matrix (Manual Calculation)
conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
for true, pred in zip(y_true, predictions):
    conf_matrix[true][pred] += 1

# Precision, Recall, and F1 Score Calculation (Manual)
precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
f1_score = 2 * (precision * recall) / (precision + recall)

# Handling NaN cases (where division by zero occurs)
precision = np.nan_to_num(precision)
recall = np.nan_to_num(recall)
f1_score = np.nan_to_num(f1_score)

# Print Results
print(f"Final Accuracy: {accuracy:.2f}")
print(f"Precision per Class: {precision}")
print(f"Recall per Class: {recall}")
print(f"F1 Score per Class: {f1_score}")
print(f"Confusion Matrix:\n{conf_matrix}")
