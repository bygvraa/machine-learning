# Convolutions in Tensorflow.
# Example p. 453 in Hands on Machine Learning by Aurelien Geron

from matplotlib.image import AxesImage
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.datasets import load_sample_image

# Load sample images
china = load_sample_image("china.jpg") / 255  # Normalize pixels to be between 0 and 1
flower = load_sample_image("flower.jpg") / 255

images = np.array([china, flower])
batch_size, height, width, channels = images.shape

# Create 2 filters, 7 x 7.
filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1  # Vertical line
filters[3, :, :, 1] = 1  # Horizontal line

# Define strides
strides = [10, 2]

fig, axes = plt.subplots(2, 3, figsize=(10, 6))

for i, stride in enumerate(strides):
    axes[i, 0].imshow(images[i])
    axes[i, 0].axis('off')

    outputs = tf.nn.conv2d(
        input=images,
        filters=filters,
        strides=stride,
        padding="SAME")
    
    for j in range(2):
        ax = axes[j, i + 1]
        ax.set_title(f'Strides: {stride}')
        ax.axis('off')
        ax.imshow(outputs[j, :, :, 1])  # Plot feature map

plt.show()
