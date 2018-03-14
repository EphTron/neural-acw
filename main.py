import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
            output = self.calc_output(inputs)
            error = target - output
            # error = math.sqrt(pow(target - output, 2))
            for idx, w in enumerate(self.weights):
                self.weights[idx] = w + self.learning_rate * error * inputs[idx]
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

    # create perceptron with step function
    # perceptron = Perceptron(3, step_func, learning_rate=0.05)
    perceptron = Perceptron(4, sigmoid, learning_rate=0.09)

    # show calculated output of perceptron after training
    graphs = {'average-error': [], 'output': [], 'delta-error': [], 'o-before': [], 'e-before': [], 'o-after': [],
              'e-after': []}
    for idx, val in enumerate(data):
        input_vec = create_input_list(data, 4, idx)
        value = perceptron.calc_output(input_vec)
        graphs['o-before'].append(value)
        graphs['e-before'].append(abs(value - data[idx]))

    # setup parameters and train perceptron
    epoch_counter, epochs, error, running = 0, 3, 1, True
    while epoch_counter < epochs and running:
        indices = list(range(len(data)))
        random.shuffle(indices)
        iter_count = 0
        error_sum = 0
        for idx in indices:
            input_vec = create_input_list(data, 4, idx)
            value, error = perceptron.train(input_vec, data[idx])
            # print("Iteration", iter_count, error, running)
            error_sum += abs(error)
            # if abs(error) < 0.0000001:
            #     running = False
            #     break
            iter_count += 1
        graphs['average-error'].append(error_sum/len(data))
        print('Epoch', epoch_counter)
        epoch_counter += 1

    # show calculated output of perceptron after training
    for idx, val in enumerate(data):
        input_vec = create_input_list(data, 4, idx)
        value = perceptron.calc_output(input_vec)
        graphs['o-after'].append(value)
        graphs['e-after'].append(abs(value - data[idx]))

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.title.set_text('Perceptron Output before Training')
    ax1.set(xlabel='Samples of track.xls', ylabel='Predicted Position')
    ax1.grid()
    ax1.plot(graphs['o-before'], 'r')
    # ax1.plot(error_unlearned, 'r')

    ax2.title.set_text('Perceptron Output after Training (' + str(epochs) + ' Epochs)')
    ax2.set(xlabel='Samples of track.xls', ylabel='Predicted Position')
    ax2.grid()
    ax2.plot(graphs['o-after'], 'b')
    # ax2.plot(error_trained, 'b')

    plt.figure(2)
    plt.plot(graphs['average-error'])

    plt.show()


if __name__ == "__main__":
    main()
