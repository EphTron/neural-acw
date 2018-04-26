import numpy as np
import math
import random


def sigmoid(input_value):
    result = [1.0 / (1.0 + math.exp(-i)) for i in np.nditer(input_value)]
    return np.asarray(result)


def step_func(input_value):
    return 0 if input_value > 0 else 1


class Network:
    def __init__(self, activation_function, input_units=2, hidden_unit_vec=[2], output_units=1):
        structure = [input_units] + hidden_unit_vec + [output_units]
        self.layers = len(structure)
        print('Layer Count: ', self.layers)
        self.weights = [np.random.rand(structure[idx + 1], structure[idx]) for idx in range(len(structure) - 1)]
        self.biases = [random.uniform(-1.0, 1.0) for _ in range(len(self.weights))]
        self.inputs = [np.zeros((1, structure[idx])) for idx in range(self.layers - 1)]
        self.outputs = [np.ones((1, structure[idx])) for idx in range(self.layers - 1)]
        self.activation_function = activation_function

    def set_weights(self, weights):
        """
        Set weights of network
        :param weights: list
        :return: void
        """
        self.weights = weights

    def set_biases(self, biases):
        """
        Set weights of network
        :param biases: list
        :return: void
        """
        if len(biases) == len(self.biases):
            self.biases = biases
        else:
            raise ValueError('Passed list length does not match.')

    def make_decision(self, input_vec):
        """
        Calculate output of network
        :param input_vec: list  # length of inputs must be equal to perceptron inputs
        :return: float          # return output
        """
        i = np.asarray(input_vec)
        for idx in range(0, len(self.weights)):
            self.inputs[idx] = i
            x = np.dot(self.weights[idx], i) + self.biases[idx]
            o = sigmoid(x)
            self.outputs[idx] = o
            i = o
        return i

    def back_progagate(self, outputs, targets, learning_rate=0.1):
        weight_updates = []
        o = np.asarray(outputs)
        t = np.asarray(targets)
        errors = np.asarray(t - o)
        print('errors ', errors)
        error_sum = np.sum(0.5 * (errors ** 2))
        # error_sum = np.sum(delta_outputs**2)  #
        print('error sum', error_sum)
        # derivative
        output_with = self.outputs[1] * (1 - self.outputs[1])
        print('output k ', output_with)
        print('output j', self.outputs[0])
        print(self.weights[0].shape)
        # print('all =', errors * output_with * self.inputs[1])
        # test = errors * output_with * self.inputs[1]
        # print('internet', test)
        # print(' test val', 0.4 - 0.1 * (test[0]))

        # for idx in reversed(range(len(self.weights))):
        #     old_weight = self.weights[idx].T
        #     derivative = self.outputs[idx] * (1 - self.outputs[idx])
        #     print(self.inputs[idx])
        #     gradient_of_error = -errors * derivative * self.inputs[idx]
        #     print(gradient_of_error)
        #     weight_updates = old_weight - learning_rate * gradient_of_error
        #     print('new weights', weight_updates)
        print('#######')

        for idx in reversed(range(len(self.weights))):
            print('errors ', errors)
            updated_weights = np.zeros(self.weights[idx].shape)
            for (k, j), value in np.ndenumerate(self.weights[idx]):
                print('j=', j, 'k=', k)
                derivative = self.outputs[idx][k] * (1 - self.outputs[idx][k])
                gradient_of_error = -errors[k] * derivative * self.inputs[idx][j]
                updated_weights[k, j] = value - learning_rate * gradient_of_error
            weight_updates.append(updated_weights)
            errors = np.dot(self.weights[idx].T, errors)
        print(weight_updates[1])
        print(weight_updates[0])
        #
        # print(self.inputs[idx])
        # gradient_of_error = -errors * derivative * self.inputs[idx]
        # print(gradient_of_error)
        # weight_updates = old_weight - learning_rate * gradient_of_error
        # print('new weights', weight_updates)
        # print('delta ', delta_outputs, idx)
        # print('inputs', self.inputs[idx])
        #
        # delta = delta_outputs * (sigmoid(self.inputs[idx]) * (1 - sigmoid(self.inputs[idx])))
        # print('delta ', delta)
        # delta_outputs = np.dot(self.weights[idx], delta)
        # print('d out ', delta_outputs)
        # Wjk = Wjk - learning_rate * (e_at_node_k * (output_at_node_k * (1 - output_at_node_k)) * output_at_node_j)

        # e_fraction =


def main():
    # read data
    file = open('Set46.txt', 'r')
    data = [l.split() for l in file.readlines()]

    net = Network(sigmoid, 2, [2], 2)
    w1 = [[0.15, 0.2],
          [0.25, 0.3]]
    w2 = [[0.4, 0.45],
          [0.5, 0.55]]
    # test with calculated values
    # w1 = [[0.9, 0.3, 0.4], [0.2, 0.8, 0.2], [0.1, 0.5, 0.6]]
    # w2 = [[0.3, 0.7, 0.5], [0.6, 0.5, 0.2], [0.8, 0.1, 0.9]]
    # # w1 = [[0.9, 0.2, 0.1], [0.2, 0.8, 0.5], [0.4, 0.2, 0.6]]
    # # w2 = [[0.3, 0.6, 0.8], [0.7, 0.5, 0.1], [0.5, 0.2, 0.9]]
    net.set_weights([np.asarray(w1), np.asarray(w2)])
    net.set_biases([0.35, 0.6])
    out = net.make_decision([0.05, 0.1])
    # out = net.make_decision([0.9, 0.1, 0.8])
    print('Inputs ', net.inputs)
    print('Outputs ', net.outputs)

    net.back_progagate(out, [0.01, 0.99], 0.5)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import random


def sigmoid(input_value):
    return 1.0 / (1.0 + math.exp(-input_value))


def step_func(input_value):
    return 1 if input_value > 0 else 0


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

    # setup parameters
    input_count, learning_rate = 4, 0.05
    activation = step_func  # sigmoid

    # setup perceptron
    weights = [random.uniform(0.0, 1.0) for _ in range(input_count)]
    bias = random.uniform(-1.0, 1.0)
    graphs = {'error': [], 'output': [], 'delta-error': [], 'before': [], 'after': []}

    # output before training
    for sample_idx in range(len(data)):
        inputs = create_input_list(data, input_count, sample_idx)
        x = np.dot(np.array(weights), np.array(inputs)) + bias
        graphs['before'].append(activation(x))

    # training
    for sample_idx in range(len(data)):
        inputs = create_input_list(data, input_count, sample_idx)

        # calc output
        x = np.dot(np.array(weights), np.array(inputs)) + bias
        output = activation(x)

        # adjusting weights
        error = data[sample_idx] - output
        for w_idx, w in enumerate(weights):
            weights[w_idx] = w + learning_rate * error * inputs[w_idx]

        # safe calculations for later plotting
        graphs['error'].append(error)
        graphs['output'].append(output)
        graphs['delta-error'].append(abs(output - data[sample_idx]))

    # output after training
    for sample_idx in range(len(data)):
        inputs = create_input_list(data, input_count, sample_idx)
        x = np.dot(np.array(weights), np.array(inputs)) + bias
        graphs['after'].append(activation(x))

    # setup plots for output
    nn_label = 'NN structure: ' + str(self.strct)
    lr_label = 'learn rate = ' + str(learn_rate)
    mo_label = 'momentum = ' + str(momentum)

    f1, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.title.set_text('ACW I: Perceptron Output before Training')
    ax1.set(xlabel='Samples of track.xls', ylabel='Predicted Position')
    ax1.grid()
    nn = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=nn_label)
    lr = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=lr_label)
    mo = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=mo_label)
    ax1.plot(graphs['before'], 'r')
    ax2.title.set_text('ACW I: Perceptron Output after Training')
    ax2.set(xlabel='Samples of track.xls', ylabel='Predicted Position')
    ax2.grid()
    ax2.plot(graphs['after'], 'b')

    # setup plots for errors
    f1, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.title.set_text('ACW I: Perceptron Delta Error')
    ax1.set(xlabel='Samples of track.xls', ylabel='Deviation (Delta Error)')
    ax1.grid()
    ax1.plot(graphs['delta-error'], 'r')
    ax2.title.set_text('ACW I: Perceptron Error Change during Training')
    ax2.set(xlabel='Samples of track.xls', ylabel='Predicted Position')
    ax2.grid()
    ax2.plot(graphs['error'], 'b')

    plt.show()

    # Questions:
    # - Update function correct?
    # - Better way to plot errors (average)
    # - Overfitting after too many training epochs


if __name__ == "__main__":
    main()
