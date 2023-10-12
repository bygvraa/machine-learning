from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

# Array of points with a classification
X = np.array(np.matrix(
    '34.62365962451697,78.0246928153624;30.28671076822607,43.89499752400101; 35.84740876993872,'
    '72.90219802708364;60.18259938620976,86.30855209546826;79.0327360507101,75.3443764369103;45.08327747668339,'
    '56.3163717815305; 61.10666453684766,96.51142588489624; 75.02474556738889,46.55401354116538; 76.09878670226257,'
    '87.42056971926803;84.43281996120035,43.53339331072109; 95.86155507093572,38.22527805795094; 75.01365838958247,'
    '30.60326323428011; 82.30705337399482,76.48196330235604; 69.36458875970939,97.71869196188608; 39.53833914367223,'
    '76.03681085115882; 53.9710521485623,89.20735013750205; 69.07014406283025,52.74046973016765; 67.94685547711617,'
    '46.67857410673128; 70.66150955499435,92.92713789364831; 76.97878372747498,47.57596364975532;67.37202754570876,'
    '42.83843832029179'))
y = np.array(np.matrix('0;0;0;1;1;0;1;1;1;1;0;0;1;1;0;1;1;0;1;1;0'))[:, 0]

# Split the data points into two arrays based on their classification
pos = np.where(y == 1)
neg = np.where(y == 0)

# Create an array of regularization parameter values
C = [0.0001, 0.001, 0.01, 0.1, 1]

# Create a figure to hold the subplots
fig, axs = plt.subplots(1, 5, figsize=(15, 3))  # Create 1 row with 5 subplots

for i, C in enumerate(C):
    # Create a subplot
    ax = axs[i]

    # Create a logistic regression model with the current C value
    logreg = linear_model.LogisticRegression(C=C)
    model = logreg.fit(X, y)

    # Plot the data points
    ax.plot(X[pos[0], 0], X[pos[0], 1], 'ro', label='Class 1')  # Red dots for class 1
    ax.plot(X[neg[0], 0], X[neg[0], 1], 'bo', label='Class 0')  # Blue dots for class 0

    # Set the x and y axis limits based on the data range
    ax.set_xlim(min(X[:, 0]), max(X[:, 0]))
    ax.set_ylim(min(X[:, 1]), max(X[:, 1]))

    # Calculate the decision boundary
    xx = np.linspace(0, 100)
    yy = - (model.coef_[0, 0] / model.coef_[0, 1]) * xx - (model.intercept_[0] / model.coef_[0, 1])

    # Plot the decision boundary as a black line
    ax.plot(xx, yy, 'k-', label=f'C={C}')

    # Calculate and display accuracy score
    accuracy = model.score(X, y)
    ax.set_title(f'C: {C}\nAccuracy: {accuracy:.2f}')

plt.tight_layout()
plt.show()

# # model.coef_[0,0]*x + model.coef_[0,1]*y + model.intercept_[0] = 0
# # y = - ( model.coef_[0,0]*x +  model.intercept_[0]) / model.coef_[0,1]
