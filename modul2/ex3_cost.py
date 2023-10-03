import numpy as np


# Define cost function to calculate the error of linear regression model
def cost(a, b, X, y):
    # Evaluate half MSE (Mean square error)

    # Calculate the number of data points
    m = len(y)

    # Calculate the predicted values using the linear equation
    predictions = a + b * X

    # Calculate the error by subtracting the actual values (y) from the predictions
    error = predictions - y

    # Calculate the cost using the Mean Square Error (MSE) formula
    J = np.sum(error ** 2) / (2 * m)

    return J


# Generate some random data to test the linear regression
# X represents input features, and y represents the actual output
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Define ranges for the 'a' and 'b' parameters of the linear regression model
# These ranges are used to search for the best 'a' and 'b' values
ainterval = np.arange(1, 10, 0.01)
binterval = np.arange(0.5, 5, 0.01)

# Make variables to store the lowest cost and the best 'a' and 'b' values
low = cost(0, 0, X, y)
bestatheta = 0
bestbtheta = 0

# Optimize the linear regression model by trying different 'a' and 'b' values
for atheta in ainterval:
    for btheta in binterval:
        # Calculate the cost for the current 'a' and 'b' values
        currentcost = cost(atheta, btheta, X, y)

        # print("x, y, cost: %f:%f:%f" % (atheta, btheta, currents))

        # Check if the current cost is lower than the lowest cost found so far
        if currentcost < low:
            # If true, update the lowest cost and the best 'a' and 'b' values
            low = currentcost
            bestatheta = atheta
            bestbtheta = btheta

# Print the best 'a' and 'b' values that result in the lowest cost (best-fit line)
print("Best 'a': %f" % bestatheta)
print("Best 'b': %f" % bestbtheta)
print("Cost: %f" % low)
