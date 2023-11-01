# CODE FROM NOTEBOOK, Exercise 6, ML WEEK 8
# Deep Learning Neural Network for the CIFAR-10 dataset
# Good shot at convolutions

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from keras.layers import Input, Flatten, Dense, Conv2D, BatchNormalization, LeakyReLU, Dropout, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical


NUM_CLASSES = 10
CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])


(X_train, y_train), (X_test, y_test) = cifar10.load_data()


''' Data preparation '''

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

# Test that it is scaled
X_train.shape
X_train[5400, 17, 9, 1]
X_test.shape


''' Model setup '''

input_layer = Input((32, 32, 3))

x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(input_layer)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Flatten()(x)

x = Dense(128)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(rate=0.5)(x)

x = Dense(NUM_CLASSES)(x)
output_layer = Activation('softmax')(x)

model = Model(input_layer, output_layer)
model.summary()

opt = Adam(learning_rate=0.0005)
model.compile(loss='categorical_crossentropy',
              optimizer=opt, metrics=['accuracy'])


''' Model training '''

model.fit(X_train, y_train, batch_size=32, epochs=10,
          shuffle=True, validation_data=(X_test, y_test))
y_pred = model.predict(X_test)


''' Model evaluation '''

model.evaluate(X_test, y_test, batch_size=1000)


''' Plot result '''

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
