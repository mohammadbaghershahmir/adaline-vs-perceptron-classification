import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import Perceptron
from numpy.linalg import inv

df = pd.read_excel("D:/Pattern Recognition/MiniProject3/Project2_dataset.xlsx")
feature1_data = df.iloc[:, 0].values
feature2_data = df.iloc[:, 1].values
class_data = df.iloc[:, 2].values
X = np.column_stack((feature1_data, feature2_data))
X_train, X_test, y_train, y_test = train_test_split(X, class_data, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Use SGDClassifier instead of SGDRegressor
adaline_model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
adaline_model.fit(X_train_scaled, y_train)
y_pred_adaline = adaline_model.predict(X_test_scaled)
# Scatter plot of data points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, edgecolors='k', marker='o', label='Actual')
# Create a meshgrid to plot the decision boundary
h = .02
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Predict the labels for each point in the meshgrid
Z = adaline_model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.title('Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
perceptron_model = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
perceptron_model.fit(X_train_scaled, y_train)
y_pred_perceptron = perceptron_model.predict(X_test_scaled)
# Plot decision boundary for Perceptron
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, edgecolors='k', marker='o', label='Actual')
# Create a meshgrid to plot the decision boundary
h = .02
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict the labels for each point in the meshgrid
Z_perceptron = perceptron_model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
# Put the result into a color plot
Z_perceptron = Z_perceptron.reshape(xx.shape)
plt.contourf(xx, yy, Z_perceptron, cmap=plt.cm.Paired, alpha=0.8)
plt.title('Perceptron Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
# Compare Adaline and Perceptron
print('\nComparison of Adaline and Perceptron:')
print('-------------------------------------')

# Accuracy
accuracy_adaline = accuracy_score(y_test, y_pred_adaline)
accuracy_perceptron = accuracy_score(y_test, y_pred_perceptron)
print('Adaline Accuracy:', accuracy_adaline)
print('Perceptron Accuracy:', accuracy_perceptron)
# Other Metrics
report_adaline = classification_report(y_test, y_pred_adaline)
report_perceptron = classification_report(y_test, y_pred_perceptron)
print('\nAdaline Classification Report:')
print(report_adaline)
print('\nPerceptron Classification Report:')
print(report_perceptron)
X = np.column_stack((feature1_data, feature2_data))
X_design = np.column_stack((np.ones(len(X)), X))
print(X_design)
y = class_data
weights = inv(X_design.T.dot(X_design)).dot(X_design.T).dot(y)
print(weights)
w0, w1, w2 = weights
# Scatter plot of data points
plt.scatter(feature1_data, feature2_data, c=class_data, cmap=plt.cm.Paired, edgecolors='k', marker='o', label='Actual')
# Create a meshgrid to plot the decision boundary
h = .02
x_min, x_max = feature1_data.min() - 1, feature1_data.max() + 1
y_min, y_max = feature2_data.min() - 1, feature2_data.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Decision boundary equation: w0 + w1*x + w2*y = 0
decision_boundary = - (w0*2.3 + w1 * xx + w2 * yy) / w2
# Plot the decision boundary
plt.contour(xx, yy, decision_boundary, levels=[0], linewidths=2, colors='red', label='Decision Boundary')
# Set labels and legend
plt.title('Decision Boundary With Analytic')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
# Show the plot
plt.show()