# CODE FROM NOTEBOOK, Exercise 4, ML WEEK 8
# Deep Learning Neural Network for the CIFAR-10 dataset

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from keras.layers import Input, Flatten, Dense, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical


NUM_CLASSES = 10
CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])


# Load CIFAR-10 dataset, which contains color images in 10 categories
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


''' Data preparation '''

# Normalize the images to a range of 0 to 1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convert target labels to one-hot encoding
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

# Test that it is scaled
X_train.shape
X_train[5400, 17, 9, 1]
X_test.shape


''' Model setup '''

input_layer = Input((32, 32, 3))  # 3 colors channels

x = Flatten()(input_layer)  # Flatten input from 2D to 1D

# In a Dense layer, every neuron (node) in the current layer is connected to every neuron in the previous layer.
# This means that each neuron in the Dense layer receives input from all the neurons in the previous layer.
x = Dense(200, activation='relu')(x)
x = Dense(150, activation='relu')(x)

output_layer = Dense(NUM_CLASSES, activation='softmax')(x)

# Create the neural network model
model = Model(input_layer, output_layer)
model.summary()

# Configure and compile the model
opt = Adam(learning_rate=0.0005)
model.compile(loss='categorical_crossentropy',
              optimizer=opt, metrics=['accuracy'])


''' Model training '''

model.fit(X_train, y_train, batch_size=32, epochs=10, shuffle=True)
y_pred = model.predict(X_test)


''' Model evaluation '''

model.evaluate(X_test, y_test)


''' Plot result '''

# Randomly choose test samples for visualization
n_to_show = len(CLASSES)
indices = np.random.choice(range(len(X_test)), n_to_show)

# print(indices)

fig, axes = plt.subplots(1, n_to_show, figsize=(16, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, idx in enumerate(indices):
    ax = axes[i]
    ax.imshow(X_test[idx])
    ax.axis('off')
    ax.text(0.5, -0.35, f'Pred: {CLASSES[np.argmax(y_pred[idx])]}',
            fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.5, -0.7, f'Act: {CLASSES[np.argmax(y_test[idx])]}',
            fontsize=10, ha='center', transform=ax.transAxes)

plt.show()
