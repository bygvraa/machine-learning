from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

# Array of data points ('X') and their corresponding classifications ('y')
X = np.array(np.matrix(
    '4,450;5,600;6,700;4.5,550;4.9,500;5,650;5.5,500; 5.25,525; 4.25,625; 4.75,575'))
y = np.array(np.matrix('0;1;1;0;0;1;0;1;1;1'))[:, 0]

# Split the data points into two arrays based on their classification
pos = np.where(y == 1)
neg = np.where(y == 0)

# Create a scatter plot of the data points
plt.plot(X[pos[0], 0], X[pos[0], 1], 'ro')  # Red for class 1
plt.plot(X[neg[0], 0], X[neg[0], 1], 'bo')  # Blue for class 0

# Set the x and y axis limits based on the data range
plt.xlim(min(X[:, 0]), max(X[:, 0]))
plt.ylim(min(X[:, 1]), max(X[:, 1]))

# Create a logistic regression model with regularization parameter
# Regularization parameter C (low = strong regularization, high = weak regularization)
logreg = linear_model.LogisticRegression(C=1000)

model = logreg.fit(X, y)

# Create a range of x values ('xx') for plotting the decision boundary
xx = np.linspace(4, 6)

# Calculate the decision boundary ('yy') for the logistic regression model
theta_0 = model.coef_[0, 0]
theta_1 = model.coef_[0, 1]
theta_2 = model.intercept_[0]

a = -(theta_0 / theta_1)
b = -(theta_2 / theta_1)

# y = ax + b
yy = a * xx + b

# Plot the decision boundary as a black line
plt.plot(xx, yy, 'k-', label=f'y = {a:.2f}x + {b:.2f}')
plt.legend()
plt.show()
