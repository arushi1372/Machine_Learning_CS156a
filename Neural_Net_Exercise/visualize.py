################################################################################
# CS 156a Bonus Exercise
# Author: Aadyot Bhatnagar
# Last modified: October 27, 2018
# Description: A script to visualize some examples from the MNIST dataset of
#              handwritten digits
################################################################################

import matplotlib.pyplot as plt
from keras.datasets import mnist

if __name__ == '__main__':

    train, val = mnist.load_data()
    nrow, ncol = 3, 5

    for data, label, kind in [(*train, 'Training'), (*val, 'Validation')]:
        for i in range(nrow):
            for j in range(ncol):
                idx = i * ncol + j
                plt.subplot(nrow, ncol, idx + 1)
                plt.imshow(data[idx], cmap='gray')
                plt.title('Label: %d' % label[idx])
                plt.axis('off')

        plt.suptitle('Example %s Images' % kind)
        plt.savefig('%s_example.png' % kind.lower())
        plt.show()
