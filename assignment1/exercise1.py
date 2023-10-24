import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary(classifier, X):
    h = .01  # Stepsize in the mesh

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict on meshgrid
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, levels=[-1, 0, 1],
                 colors=['darkblue', 'yellow'], alpha=.1)
    plt.contour(xx, yy, Z, cmap='viridis', alpha=1, linewidths=.2)
