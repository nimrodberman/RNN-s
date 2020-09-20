import numpy as np


class RNN:

    def __init__(self, input_size, output_size, hidden_state_size):
        # initialize biases
        self.bh = np.zeros((hidden_state_size, 1))
        self.by = np.zeros((output_size, 1))

        # initialize weights
        self.Wxh = np.random.randn(hidden_state_size, input_size) / 1000
        self.Whh = np.random.randn(hidden_state_size, hidden_state_size) / 1000
        self.Why = np.random.randn(output_size, hidden_state_size) / 1000

    # we feed the network with our series data inputs
    def forward(self, inputs):

        # array for the series outputs
        outputs = []

        h = np.zeros((self.Whh.shape[0], 1))

        self.last_inputs = inputs
        self.last_hs = {0: h}

        # run the recurrent calculation. Keep each yt at an output array.
        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.last_hs[i+1] = h

            # compute the output and insert it
            outputs.append(self.Why @ h + self.by)

        """ we will return only the last output but we can return 
        "Many to many" only by return the list instead of its last element """
        return h, outputs[len(outputs)-1]

    def backpropagation(self, dl_dy):
        # learn rate
        learn_rate = 1e-1

        # calculate the partial derivative of Why and By
        dl_dWhy = dl_dy @ self.last_hs[len(self.last_inputs)].T
        dl_dby = dl_dy

        # initiate matrices
        dl_dWxh = np.zeros(self.Wxh.shape)
        dl_dWhh = np.zeros(self.Whh.shape)
        dl_dbh = np.zeros(self.bh.shape)

        # calculate dl_dhn
        dl_dh = self.Why.T @ dl_dy

        # calculate from the back to the start though time
        for t in reversed(range(len(self.last_inputs))):
            common = ((1 - self.last_hs[t+1] ** 2) * dl_dh)

            dl_dbh += common
            dl_dWhh += common @ self.last_hs[t].T
            dl_dWxh += common @ self.last_inputs[t].T

            # next hidden state derivative
            dl_dh = self.Whh @ common

        # update weights and biases with gradient decent
        self.bh -= dl_dbh
        self.by -= dl_dby

        self.Wxh -= learn_rate * dl_dWxh
        self.Whh -= learn_rate * dl_dWhh
        self.Why -= learn_rate * dl_dWhy























