import numpy as np
import math
import random
import matplotlib.pyplot as plt
from math import floor


def split_to_test_and_training(data, train_size=0.8):
    random.shuffle(data)
    split_idx = floor(len(data) * train_size)
    trainings_set = data[:split_idx]
    test_set = data[split_idx:]
    return trainings_set, test_set


def sigmoid(input_value):
    result = [1.0 / (1.0 + math.exp(-i)) for i in np.nditer(input_value)]
    return np.asarray(result)


def step_func(input_value):
    return 0 if input_value > 0 else 1


class Network:
    def __init__(self, activation_function, input_units=2, hidden_unit_vec=[2], output_units=1):
        self.strct = [input_units] + hidden_unit_vec + [output_units]
        self.layers = len(self.strct)
        print('Layer Count: ', self.layers)
        self.weights = [np.random.normal(0.0, 0.7, (self.strct[idx + 1], self.strct[idx])) for idx in
                        range(len(self.strct) - 1)]
        # self.weights = [np.random.rand(self.strct[idx + 1], self.strct[idx]) for idx in range(len(self.strct) - 1)]
        self.last_update = [np.zeros((self.strct[idx + 1], self.strct[idx])) for idx in range(len(self.strct) - 1)]
        self.biases = [random.uniform(-1.0, 1.0) for _ in range(len(self.weights))]
        self.inputs = [np.zeros((1, self.strct[idx])) for idx in range(self.layers - 1)]
        self.outputs = [np.ones((1, self.strct[idx])) for idx in range(self.layers - 1)]
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

    def back_progagate(self, inputs, targets, learning_rate, momentum):
        weight_updates = []
        o = self.make_decision(inputs)
        t = np.asarray(targets)
        errors = np.asarray(t - o)
        error_sum = np.sum(errors ** 2)
        # print('error sum', error_sum)
        for idx in reversed(range(len(self.weights))):
            new_weights = np.zeros(self.weights[idx].shape)
            layer_error = np.zeros(self.weights[idx].shape[1])
            for (k, j), w in np.ndenumerate(self.weights[idx]):
                derivative = self.outputs[idx][k] * (1 - self.outputs[idx][k])
                gradient = -errors[k] * derivative * self.inputs[idx][j]
                update = (learning_rate * gradient + momentum * self.last_update[idx][k, j])
                new_weights[k, j] = w - update
                layer_error[j] += errors[k] * derivative * w
                self.last_update[idx][k, j] = update
            weight_updates.append(new_weights)
            errors = layer_error
            self.weights[idx] = new_weights

        return error_sum

    def train(self, epochs, training_data, test_data, learn_rate=0.1, momentum=0.1):
        print('Start training')
        graphs = {'error': [], 'test-error': [], 'wanted-output': [], 'best': [], 'before': [], 'after': []}
        sorted_data = training_data
        for data_set in sorted_data:
            out = self.make_decision(data_set[:2])
            graphs['before'].append(out[0])
            graphs['wanted-output'].append(data_set[2])
        best_epoch = 1
        best_epoch_num = 0
        # best_epoch_weights = None

        smallest_error = 1
        for i in range(epochs):
            random.shuffle(training_data)
            total_epoch_error = 0
            for train_set in training_data:
                e = self.back_progagate(train_set[:2], train_set[2], learn_rate, momentum)
                total_epoch_error += e
                if smallest_error > e:
                    smallest_error = e
            total_error = total_epoch_error / len(training_data)
            graphs['error'].append(total_error)
            if best_epoch > total_error:
                best_epoch = total_error
                best_epoch_num = i
                # best_epoch_weights = self.weight

            test_epoch_error = 0
            for test_set in test_data:
                o = self.make_decision(test_set[:2])
                errors = np.asarray(test_set[2] - o)
                error_sum = np.sum(errors ** 2)
                test_epoch_error += error_sum
            graphs['test-error'].append(test_epoch_error / len(test_data))

            graphs['best'].append(best_epoch)
            print('Epoch ', i, ' Best: ', best_epoch, ' Error:', total_error)

        print('End training. Best epoch', best_epoch_num, ' Error: ', best_epoch)  # ,'\nWeights: ', best_epoch_weights)

        for data_set in sorted_data:
            out = self.make_decision(data_set[:2])
            graphs['after'].append(out[0])

        f1, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.title.set_text('ACW II: Neural Network before Training')
        ax1.set(xlabel='Set46 Samples', ylabel='Target acquired')
        ax1.grid()
        ax1.plot(graphs['wanted-output'], 'go')
        ax1.plot(graphs['before'], 'ro')
        ax2.title.set_text('ACW II: Neural Network after Training')
        ax2.set(xlabel='Set46 Samples', ylabel='Target acquired')
        ax2.grid()
        ax2.plot(graphs['wanted-output'], 'go')
        ax2.plot(graphs['after'], 'bo')

        # setup plots for errors
        f1, (ax1) = plt.subplots(1, 1, sharey=True)
        ax1.title.set_text('ACW II: Neural Network Average Error Improvement')
        ax1.set(xlabel='Epochs', ylabel='Average MSE')
        ax1.grid()
        ax1.plot(graphs['test-error'], 'b')
        ax1.plot(graphs['best'], 'g')
        ax1.plot(graphs['error'], 'r')
        # ax2.title.set_text('ACW II: Perceptron Error Change during Training')
        # ax2.set(xlabel='Samples of track.xls', ylabel='Predicted Position')
        # ax2.grid()
        # ax2.plot(graphs['error'], 'b')


def main():
    # read data
    file = open('Set46.txt', 'r')
    data = [[float(w) for w in l.split()] for l in file.readlines()]
    print('Loaded ', len(data), 'data sets.')
    random.shuffle(data)
    train_set, test_set = split_to_test_and_training(data, 0.7)

    # # Test LR 1 from 0.2435 to 0.2126 in 500 epochs
    # Note oscillates
    # net = Network(sigmoid, 2, [5], 1)
    # net.train(500, train_set, test_set, 0.75, 0.00)

    # # Test LR 2 from 0.2465 to 0.1680 in 500 epochs
    # net = Network(sigmoid, 2, [5], 1)
    # net.train(500, train_set, test_set, 0.15, 0.00)

    # # Test LR 3 from 0.2611 to 0.2349 in 250 epochs
    # net = Network(sigmoid, 2, [5], 1)
    # net.train(250, train_set, test_set, 0.10, 0.00)

    # # Test LR 4 from 0.2611 to 0.2349 in 250 epochs
    # net = Network(sigmoid, 2, [5], 1)
    # net.train(250, train_set, test_set, 0.30, 0.00)

    # # Test Nodes 1 from 0.2604 to 0.2189 in 500 epochs
    # Note oscillates
    # net = Network(sigmoid, 2, [10], 1)
    # net.train(500, train_set, test_set, 0.55, 0.02)

    # # Test Nodes 2 from 0.2419 to 0.2382 in 500 epochs
    # net = Network(sigmoid, 2, [2], 1)
    # net.train(500, train_set, test_set, 0.5, 0.02)

    # Test Momentum 1 from 0.2493 to 0.0814 in 250 epochs
    # net = Network(sigmoid, 2, [10], 1)
    # net.train(250, train_set, test_set, 0.2, 0.1)

    # # Test Momentum 2 from 0.2533 to 0.1280 in 250 epochs
    # net = Network(sigmoid, 2, [10], 1)
    # net.train(250, train_set, test_set, 0.2, 0.2)

    # # Test from 0.2611 to 0.2349 in 250 epochs
    net = Network(sigmoid, 2, [15], 1)
    net.train(1200, train_set, test_set, 0.18, 0.11)
    plt.show()


if __name__ == "__main__":
    main()
