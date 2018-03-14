import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random


def sigmoid(input_value):
    return 1.0 / (1.0 + math.exp(-input_value))


def step_func(input_value):
    return 0 if input_value > 0 else 1


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
    create_input_list(data, 3, 1)

    # setup parameters
    input_count, target, learning_rate = [], 2, 0.05

    # setup perceptron
    weights = [random.uniform(0.0, 1.0) for _ in range(len(input_count))]
    bias = random.uniform(-1.0, 1.0)
    graphs = {'error': [], 'output': [], 'delta-error': []}
    output_1, output_2 = [], []

    for sample_idx in range(len(data)):
        inputs = create_input_list(data, input_count, sample_idx)
        x = np.dot(np.array(weights), np.array(inputs)) + bias
        output_1.append(sigmoid(x))

    for sample_idx in range(len(data)):
        inputs = create_input_list(data, input_count, sample_idx)

        # calc output
        x = np.dot(np.array(weights), np.array(inputs)) + bias
        output = sigmoid(x)

        # backpropagate
        error = target - output
        for w_idx, w in enumerate(weights):
            weights[w_idx] = w + learning_rate * error * inputs[w_idx] / 2

        # safe and print calculations
        graphs['error'].append(error)
        graphs['output'].append(output)
        graphs['delta-error'].append(abs(output - data[sample_idx]))

    for sample_idx in range(len(data)):
        inputs = create_input_list(data, input_count, sample_idx)
        x = np.dot(np.array(weights), np.array(inputs)) + bias
        output_2.append(sigmoid(x))

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.title.set_text('Perceptron Output before Training')
    ax1.set(xlabel='Samples of track.xls', ylabel='Predicted Position')
    ax1.grid()
    ax1.plot(output_1, 'r')

    ax2.title.set_text('Perceptron Output after Training')
    ax2.set(xlabel='Samples of track.xls', ylabel='Predicted Position')
    ax2.grid()
    ax2.plot(output_2, 'b')

    plt.show()


if __name__ == "__main__":
    main()