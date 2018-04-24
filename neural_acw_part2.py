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

    def backprogagate(self, outputs, targets, learning_rate=0.1):
        weight_updates = []
        o = np.asarray(outputs)
        t = np.asarray(targets)
        errors = np.asarray(t - o)
        error_sum = np.sum(0.5 * (errors ** 2))
        print('error sum', error_sum)
        # error_sum = np.sum(delta_outputs**2)  #

        for idx in reversed(range(len(self.weights))):
            print('errors ', errors)
            updated_weights = np.zeros(self.weights[idx].shape)
            for (k, j), value in np.ndenumerate(self.weights[idx]):
                print('j=', j, 'k=', k)
                derivative = self.outputs[idx][k] * (1 - self.outputs[idx][k])
                gradient_of_error = -errors[k] * derivative * self.inputs[idx][j]
                if idx == 0:
                    print('input',self.inputs[idx][j])
                    print('derivative', derivative)
                    print('error', -errors[k])
                    print('new weight', value)
                updated_weights[k, j] = value - learning_rate * gradient_of_error
            weight_updates.append(updated_weights)
            errors = np.dot(self.weights[idx].T, errors)
        print(weight_updates[1])
        print(weight_updates[0])

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


if __name__ == "__main__":
    main()
