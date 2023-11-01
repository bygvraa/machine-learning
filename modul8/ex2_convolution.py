# example of using a single convolutional layer
# to detect a line in some input data.
# Demo of the calculations

import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D

# define input data
data = [[0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0]]

data = np.asarray(data)
data = data.reshape(1, 8, 8, 1)

# create model

# Next, we can define a model that expects input samples to have the shape (8, 8, 1)
# and has a single hidden convolutional layer
# with a single filter with the shape of three pixels by three pixels
model = Sequential()
model.add(Conv2D(1, (3, 3), input_shape=(8, 8, 1)))

# summarize model
model.summary()

# define a vertical line detector

# The filter is initialized with random weights as part of the initialization of the model.
# We will overwrite the random weights
# and hard code our own 3×3 filter that will detect vertical lines
detector = [[[[0]], [[1]], [[0]]],
            [[[0]], [[1]], [[0]]],
            [[[0]], [[1]], [[0]]]]

weights = [np.asarray(detector), np.asarray([0.0])]

# store the weights in the model
model.set_weights(weights)

# apply filter to input data
# Next, we can apply the filter to our input image by calling the predict() function on the model.
yhat = model.predict(data)

# enumerate rows
for r in range(yhat.shape[1]):
    # print each column in the row
    print([yhat[0, r, c, 0] for c in range(yhat.shape[2])])
