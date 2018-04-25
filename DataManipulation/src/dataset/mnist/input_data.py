from tensorflow.examples.tutorials.mnist import input_data
from dataset.mnist import mnist_data_dir


def get_mnist_dataset():
    return input_data.read_data_sets(mnist_data_dir, one_hot=False)

# Note: set `one_hot=True` will cause error like `Cannot feed value of shape (50, 10) for Tensor 'Placeholder_1:0', which has shape '(?,)'`
