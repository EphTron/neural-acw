import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import random


def sigmoid(input_value):
    return 1.0 / (1.0 + math.exp(-input_value))


def step_func(input_value):
    return 0 if input_value > 0 else 1


class Perceptron:
    def __init__(self, input_units, activation_function, learning_rate=0.05):
        """
        Create simple perceptron with n inputs and one output.
        Set an activation function and set learn rate
        Weights and bias get set to a random value
        :param input_units: int                 # Define the number of inputs needed
        :param activation_function: function    # Pass activation function f(x) = y
        :param learning_rate: float             # Define learning rate
        """
        self.input_units = input_units
        self.weights = [random.uniform(0.0, 1.0) for _ in range(input_units)]
        self.bias = random.uniform(-1.0, 1.0)
        self.learning_rate = learning_rate
        self.activation_function = activation_function

    def calc_output(self, inputs):
        """
        Calculate output of perceptron
        :param inputs: list  # length of inputs must be equal to perceptron inputs
        :return: float       # return output
        """
        if len(inputs) is self.input_units:
            x = np.dot(np.array(self.weights), np.array(inputs)) + self.bias
            return self.activation_function(x)

    def train(self, inputs, target):
        """
        Train perceptron with given inputs and adjust weights based on target.
        :param inputs: list   # length of inputs must be equal to perceptron inputs
        :param target: float  # wanted output of the perceptron
        :return: float        # returns difference between
        """

        if len(inputs) is self.input_units:
            # new_weights = [random.uniform(0.0, 1.0) for _ in range(input_units)]
            output = self.calc_output(inputs)
            error = target - output
            # error = math.sqrt(pow(target - output, 2))
            for idx, w in enumerate(self.weights):
                self.weights[idx] = w + self.learning_rate * error * inputs[idx]
                self.bias = self.bias + self.learning_rate * error
            return output, error


def create_input_list(data, inputs, idx):
    input_vec = []
    if inputs > 1:
        for i in reversed(range(1, inputs)):
            input_vec.append(data[idx - i]) if (idx - i >= 0) else input_vec.append(0)
    input_vec.append(data[idx])
    return input_vec


def main():
    # read data and transform to list
    xls_data = pd.ExcelFile("track.xls")
    data = [x[0] for x in pd.Series.tolist(xls_data.parse('data1'))]
    input_units, learn_rate = 4, 0.05
    activation = sigmoid
    ac_label = 'Activation f: Step'
    in_label = 'Input units: ' + str(input_units)
    lr_label = 'learn rate = ' + str(learn_rate)
    # create perceptron with step function
    perceptron = Perceptron(input_units, activation, learning_rate=learn_rate)

    # show calculated output of perceptron after training
    graphs = {'average-error': [], 'best': [], 'output': [], 'delta-error': [], 'o-before': [], 'e-before': [], 'o-after': [],
              'e-after': []}

    # assess output of data before training
    for idx, val in enumerate(data):
        input_vec = create_input_list(data, input_units, idx)
        value = perceptron.calc_output(input_vec)
        graphs['o-before'].append(value)
        graphs['e-before'].append(data[idx])

    # setup parameters and train perceptron
    best = 1
    epoch_counter, epochs, error, running = 0, 250, 1, True
    while epoch_counter < epochs and running:
        indices = list(range(len(data)))
        random.shuffle(indices)
        iter_count = 0
        error_sum = 0

        for idx in indices:
            input_vec = create_input_list(data, input_units, idx)
            value, error = perceptron.train(input_vec, data[idx])
            error_sum += abs(error)
            iter_count += 1
        graphs['average-error'].append(error_sum / len(data))
        if (error_sum / len(data)) < best:
            best = error_sum / len(data)
        graphs['best'].append(best)
        print('Epoch ', epoch_counter, ' Best ', best)
        epoch_counter += 1

    # show calculated output of perceptron after training
    for idx, val in enumerate(data):
        input_vec = create_input_list(data, input_units, idx)
        value = perceptron.calc_output(input_vec)
        graphs['o-after'].append(value)
        graphs['e-after'].append(data[idx])
        print()

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.title.set_text('Perceptron Output before Training')
    ax1.set(xlabel='Samples of track.xls', ylabel='Predicted Position')
    ax1.grid()
    ax1.plot(graphs['e-before'], 'r')
    ax1.plot(graphs['o-before'], 'b')
    ax2.title.set_text('Perceptron Output after Training (' + str(epochs) + ' Epochs)')
    ax2.set(xlabel='Samples of track.xls', ylabel='Predicted Position')
    ax2.grid()
    ac = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=ac_label)
    iu = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=in_label)
    lr = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=lr_label)
    p1, = ax2.plot(graphs['e-after'], 'r', label='Real Position')
    p2, = ax2.plot(graphs['o-after'], 'b', label='Predicted Position')
    ax2.legend([ac, iu, lr, p2, p1],
               [ac_label, in_label, lr_label, 'Predicted Position' , 'Real Position'], prop={'size': 14})

    # setup plots for errors
    f2, (ax1) = plt.subplots(1, 1, sharey=True)
    ax1.title.set_text('ACW I: Perceptron Average Error Improvement')
    ax1.set(xlabel='Epochs', ylabel='Avg. Error in Epoch')
    ax1.grid()
    ac = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=ac_label)
    iu = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=in_label)
    lr = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=lr_label)
    best_error, = ax1.plot(graphs['best'], 'g', label='Lowest NN Error')
    avg_error, = ax1.plot(graphs['average-error'], 'r', label='Avg. Error of Epoch')
    ax1.legend([ac,iu, lr, avg_error, best_error],
               [ac_label, in_label, lr_label, 'Avg. Error of Epoch', 'Lowest NN Error'], prop={'size': 14})

    plt.show()


if __name__ == "__main__":
    main()
