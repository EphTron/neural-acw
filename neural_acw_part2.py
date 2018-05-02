import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
            # bias = self.biases[idx] - learning_rate * (np.sum(errors) / len(errors))
            # self.biases[idx] = bias

        return error_sum

    def train(self, epochs, data, learn_rate=0.1, momentum=0.1):
        training_data, test_data = split_to_test_and_training(data, 0.85)
        print('Start training')
        graphs = {'error': [], 'test-error': [], 'wanted-output': [], 'best': [], 'before': [], 'after': []}
        sorted_data = training_data

        # assess fitness before training
        for data_set in sorted_data:
            out = self.make_decision(data_set[:2])
            graphs['before'].append(out[0])
            graphs['wanted-output'].append(data_set[2])
        best_epoch = 1
        best_epoch_num = 0
        best_epoch_weights = None

        # train network
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
                best_epoch_weights = self.weights

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
        self.weights = best_epoch_weights

        # assess fitness with best weights
        for data_set in sorted_data:
            out = self.make_decision(data_set[:2])
            graphs['after'].append(out[0])

        nn_label = 'NN structure: ' + str(self.strct)
        lr_label = 'learn rate = ' + str(learn_rate)
        mo_label = 'momentum = ' + str(momentum)

        f1, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.title.set_text('ACW II: Neural Network before Training')
        ax1.set(xlabel='Set46 Samples', ylabel='Target acquired')
        ax1.grid()
        nn = ax2.plot([0, 0], 'white', label=nn_label)
        lr = ax2.plot([0, 0], 'white', label=lr_label)
        mo = ax2.plot([0, 0], 'white', label=mo_label)
        ax1.plot(graphs['wanted-output'], 'go', label='Target Output')
        ax1.plot(graphs['before'], 'bo', label='Untrained NN Output')
        ax1.legend(prop={'size': 14})
        ax2.title.set_text('ACW II: Neural Network after Training (' + str(epochs) + ' Epochs)')
        ax2.set(xlabel='Set46 Samples', ylabel='Target acquired')
        ax2.grid()
        ax2.plot(graphs['wanted-output'], 'go', label='Target Output')
        ax2.plot(graphs['after'], 'bo', label='Trained NN Output')

        # setup plots for errors
        f2, (ax3) = plt.subplots(1, 1, sharey=True)
        ax3.title.set_text('ACW II: Neural Network Average Error Improvement')
        ax3.set(xlabel='Epochs', ylabel='Average MSE')
        ax3.grid()
        nn = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=nn_label)
        lr = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=lr_label)
        mo = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=mo_label)
        test_set_err_plot, = ax3.plot(graphs['test-error'], 'grey', label='Avg. Error of Test Set')
        lowest_err_plot, = ax3.plot(graphs['best'], 'g', label='Lowest NN Error')
        train_set_err_plot, = ax3.plot(graphs['error'], 'b', label='Avg. Error of Epoch')
        ax3.legend([nn, lr, mo, train_set_err_plot, test_set_err_plot, lowest_err_plot],
                   [nn_label, lr_label, mo_label, 'Avg. Error of Epoch', 'Avg. Error of Test Set', 'Lowest NN Error'],
                   prop={'size': 14})

        return graphs

def average_graphs(g1,g2,g3):
    avg_graph = dict()
    avg_graph['error'] = [sum(x) / 3 for x in zip(g1['error'], g2['error'], g3['error'])]
    # [sum(x) / 3 for x in zip([1, 1, 1], [2, 2, 2], [3, 3, 3])]
    avg_graph['test-error'] = [sum(x) / 3 for x in zip(g1['test-error'], g2['test-error'], g3['test-error'])]
    avg_graph['wanted-output'] = [sum(x) / 3 for x in zip(g1['wanted-output'], g2['wanted-output'], g3['wanted-output'])]
    avg_graph['best'] = [sum(x) / 3 for x in zip(g1['best'], g2['best'], g3['best'])]
    avg_graph['before'] = [sum(x) / 3 for x in zip(g1['before'], g2['before'], g3['before'])]
    avg_graph['after'] = [sum(x) / 3 for x in zip(g1['after'], g2['after'], g3['after'])]
    return avg_graph

def draw_graph(graphs, strct, learn_rate, momentum, epochs):
    nn_label = 'NN structure: ' + str(strct)
    lr_label = 'learn rate = ' + str(learn_rate)
    mo_label = 'momentum = ' + str(momentum)

    f1, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.title.set_text('ACW II: Neural Network before Training')
    ax1.set(xlabel='Set46 Samples', ylabel='Target acquired')
    ax1.grid()
    nn = ax2.plot([0, 0], 'white', label=nn_label)
    lr = ax2.plot([0, 0], 'white', label=lr_label)
    mo = ax2.plot([0, 0], 'white', label=mo_label)
    ax1.plot(graphs['wanted-output'], 'go', label='Target Output')
    ax1.plot(graphs['before'], 'bo', label='Untrained NN Output')
    ax1.legend(prop={'size': 14})
    ax2.title.set_text('ACW II: Neural Network after Training (' + str(epochs) + ' Epochs)')
    ax2.set(xlabel='Set46 Samples', ylabel='Target acquired')
    ax2.grid()
    ax2.plot(graphs['wanted-output'], 'go', label='Target Output')
    ax2.plot(graphs['after'], 'bo', label='Trained NN Output')

    # setup plots for errors
    f2, (ax3) = plt.subplots(1, 1, sharey=True)
    ax3.title.set_text('ACW II: Neural Network Average Error Improvement')
    ax3.set(xlabel='Epochs', ylabel='Average MSE')
    ax3.grid()
    nn = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=nn_label)
    lr = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=lr_label)
    mo = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=mo_label)
    test_set_err_plot, = ax3.plot(graphs['test-error'], 'grey', label='Avg. Error of Test Set')
    lowest_err_plot, = ax3.plot(graphs['best'], 'g', label='Lowest NN Error')
    train_set_err_plot, = ax3.plot(graphs['error'], 'b', label='Avg. Error of Epoch')
    ax3.legend([nn, lr, mo, train_set_err_plot, test_set_err_plot, lowest_err_plot],
               [nn_label, lr_label, mo_label, 'Avg. Error of Epoch', 'Avg. Error of Test Set', 'Lowest NN Error'],
               prop={'size': 14})


def main():
    # read data
    file = open('Set46.txt', 'r')
    data = [[float(w) for w in l.split()] for l in file.readlines()]
    print('Loaded ', len(data), 'data sets.')
    random.shuffle(data)

    graph1 = {'error': [1,1], 'test-error': [1,1], 'wanted-output': [1,1], 'best': [1,1], 'before': [1,1], 'after': [1,1]}
    graph2 = {'error': [2,1], 'test-error': [2,1], 'wanted-output': [2,1], 'best': [2,1], 'before': [2,1], 'after': [2,1]}
    graph3 = {'error': [3,1], 'test-error': [3,1], 'wanted-output': [3,1], 'best': [3,1], 'before': [3,1], 'after': [3,1]}
    g = average_graphs(graph1,graph2,graph3)
    print(g)

    # Test varying number of layers and nodes
    # net1 = Network(sigmoid, 2, [3], 1)
    # g1 = net1.train(250, data, 0.20, 0.005)
    # net2 = Network(sigmoid, 2, [3], 1)
    # g2 = net2.train(250, data, 0.20, 0.005)
    # net3 = Network(sigmoid, 2, [3], 1)
    # g3 = net3.train(250, data, 0.20, 0.005)
    # g = average_graphs(g1, g2, g3)
    # draw_graph(g, [2, 3, 1], 0.20, 0.005, 250)
    #
    # net_big_1 = Network(sigmoid, 2, [30], 1)
    # gb_1 = net_big_1.train(250, data, 0.20, 0.005)
    # net_big_2 = Network(sigmoid, 2, [30], 1)
    # gb_2 = net_big_2.train(250, data, 0.20, 0.005)
    # net_big_3 = Network(sigmoid, 2, [30], 1)
    # gb_3 = net_big_3.train(250, data, 0.20, 0.005)
    # g = average_graphs(gb_1, gb_2, gb_3)
    # draw_graph(g, [2, 30, 1], 0.20, 0.005, 250)

    net_two_1 = Network(sigmoid, 2, [5, 5], 1)
    gt_1 = net_two_1.train(250, data, 0.20, 0.005)
    net_two_2 = Network(sigmoid, 2, [5, 5], 1)
    gt_2 = net_two_2.train(250, data, 0.20, 0.005)
    net_two_3 = Network(sigmoid, 2, [5, 5], 1)
    gt_3 = net_two_3.train(250, data, 0.20, 0.005)
    g = average_graphs(gt_1, gt_2, gt_3)
    draw_graph(g, [2, 5, 5, 1], 0.20, 0.005, 250)

    net_big_two_1 = Network(sigmoid, 2, [20, 20], 1)
    gbt_1 = net_big_two_1.train(250, data, 0.20, 0.005)
    net_big_two_2 = Network(sigmoid, 2, [20, 20], 1)
    gbt_2 = net_big_two_2.train(250, data, 0.20, 0.005)
    net_big_two_3 = Network(sigmoid, 2, [20, 20], 1)
    gbt_3 = net_big_two_3.train(250, data, 0.20, 0.005)
    g = average_graphs(gbt_1, gbt_2, gbt_3)
    draw_graph(g, [2, 20, 20, 1], 0.20, 0.005, 250)


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

    # # Test
    # net = Network(sigmoid, 2, [15], 1)
    # net.train(250, train_set, test_set, 0.18, 0.11)

    # # Test
    # net = Network(sigmoid, 2, [15], 1)
    # net.train(250, train_set, test_set, 0.18, 0.11)

    plt.show()


if __name__ == "__main__":
    main()
