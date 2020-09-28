import numpy as np
import torch.nn as nn
import torch

class RNN(nn.Module):

    def __init__(self, input_size, output_size, hidden_state_size, layer_number):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_state_size = hidden_state_size
        self.layer_number = layer_number

        self.rnn = nn.RNN(input_size, hidden_state_size, layer_number, batch_first=True)
        self.fc = nn.Linear(hidden_state_size, output_size)

    def forward(self, inputs):

        hidden = self.init_hidden(inputs.size(0))

        # Use PyTorch RNN models to feed forward
        # Applies a multi-layer Elman RNN with \tanhtanh or \text{ReLU}ReLU non-linearity to an input sequence.
        out, hidden = self.rnn(inputs, hidden)

        # Turning last hidden layer to final output
        out = self.fc(out.contiguous().view(-1, self.hidden_state_size))

        return out, hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.layer_number, batch_size, self.hidden_state_size)
        return hidden




















