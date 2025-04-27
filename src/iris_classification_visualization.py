import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# URL of the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Load the dataset directly from the URL
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv(url, header=None, names=columns)

# Features (X) and target (y)
X = iris.iloc[:, :-1].values  # First 4 columns as features
y = iris.iloc[:, -1].values   # Last column as target

# Convert species labels to numeric values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')

# Print evaluation metrics
print("Evaluation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")

# Visualize decision boundaries for Sepal Length and Sepal Width
X_visual = iris[['sepal_length', 'sepal_width']].values
y_visual = label_encoder.fit_transform(iris['species'])

# Train model for visualization
X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(X_visual, y_visual, test_size=0.2, random_state=42)
model_vis = LogisticRegression(max_iter=200)
model_vis.fit(X_train_vis, y_train_vis)

# Plot decision boundaries
x_min, x_max = X_visual[:, 0].min() - 1, X_visual[:, 0].max() + 1
y_min, y_max = X_visual[:, 1].min() - 1, X_visual[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict on the mesh grid
Z = model_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Create the plot
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
plt.scatter(X_visual[:, 0], X_visual[:, 1], c=y_visual, edgecolor='k', cmap='viridis')
plt.title(f"Logistic Regression Decision Boundary (Accuracy: {accuracy:.2f})")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.colorbar(label="Class")
plt.show()
