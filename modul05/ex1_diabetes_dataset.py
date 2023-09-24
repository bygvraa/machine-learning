from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn import datasets

import matplotlib.pyplot as plt

# Don't plot the sex data
diabetes = datasets.load_diabetes(as_frame=True)
features = diabetes['feature_names']
features.remove('sex')

# Plot
fig, axs = plt.subplots(3, 3)
fig.suptitle('Diabetes Dataset')
for i in range(3):
    for j in range(3):
        n = j + i * 3
        feature = features[n]
        axs[i, j].scatter(diabetes['data'][feature], diabetes['target'], s=1)
        axs[i, j].set_xlabel(feature)
        axs[i, j].set_ylabel('target')

plt.tight_layout()
plt.show()

# Load the diabetes dataset
# diabetes_X,diabetes_y = datasets.load_diabetes(return_X_y = True)
diabetes = datasets.load_diabetes()

df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
print(df.head())

# All the data points have been through preprocessed through a standard scaler,
# which transforms the data so that it has a mean of 0 and a standard deviation of 1.

X = df['bmi'].values
Y = diabetes.target

plt.scatter(X, Y)
plt.xlabel('Body mass index (BMI)')
plt.ylabel('Disease progression')
plt.show()


def plot_line(w, b):
    x_values = np.linspace(X.min(), X.max(), 100)
    y_values = w*x_values + b
    plt.plot(x_values, y_values, 'r-')


w = 1500
b = 230

plt.scatter(X, Y)
plot_line(w, b)
plt.show()


lin_reg = LinearRegression()

X = X.reshape(-1, 1)
lin_reg.fit(X, Y)

w = lin_reg.coef_[0]
b = lin_reg.intercept_

plt.scatter(X, Y)
plot_line(w, b)
plt.show()
