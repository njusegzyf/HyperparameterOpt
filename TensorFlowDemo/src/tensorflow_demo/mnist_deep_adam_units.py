"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from dataset.mnist import get_mnist_dataset

from tensorflow_demo.mnist_deep_adam import run_with_specified_args


def main(_):
    # Import data
    mnist = get_mnist_dataset()

    for units in range(1800, 2401, 100):  # range(200, 1601, 100):
        print('-' * 80)
        print('\n units {0} \n'.format(units))
        run_with_specified_args(mnist, units=units, isLogToFile=False)


if __name__ == '__main__':
    tf.app.run(main=main)
