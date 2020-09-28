# TODO: Add small batches training method
from Data import train_data, test_data, all_data
import numpy as np
import random
from RNN import RNN
import Data
import torch.nn as nn
import torch

""" info: encode characters array to one vector array that represent input sequence.
    input: [int] sequence , int dict_size , int seq_len, int batch_size
    output: [dict_size*1 vector]"""


def oneHotVector(sequence, dict_size, seq_len, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)

    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u, sequence[i][u]] = 1
    return features


def appendPadding(sentences, max_len):
    for sentence in sentences:
        for i in range(len(sentences)):
            while len(sentences[i]) < max_len:
                sentences[i] += ' '


def index2words(vocab):
    tmp = {}
    i = 0
    for item in vocab:
        tmp[i] = item
        i = i + 1
    return tmp


def word2index(vocab):
    tmp = {}
    i = 0
    for item in vocab:
        tmp[item] = i
        i = i + 1
    return tmp


""" extract all letters from the sentences and creates a letters vocabulary """


def createVocab(sentences):
    letters = set(''.join(sentences))
    return letters


"""takes list of words and return a list of vectors each is an hot-vector """


def wordsToVector(words):
    vectors = []
    for word in words:
        vector = np.zeros((vocabulary_size, 1))
        vector[word_to_index[word]] = 1
        vectors.append(vector)
    return vectors


"""" ------------- Main - data preparation -------------------- """
# upload the data
Data.uploadData(1000)
# create letters vocabulary
vocabulary = createVocab(train_data)
# get the vocabulary size
vocabulary_size = len(vocabulary)

# build word to index and index to word dictionaries
index_to_word = index2words(vocabulary)
word_to_index = word2index(vocabulary)

# turn all sentences to the same length with padding
max_length = len(max(train_data, key=len))
appendPadding(train_data, max_length)

# create input and target sequences
input_seq = []
target_seq = []
for i in range(len(train_data)):
    # Remove last character for input sequence
    input_seq.append(train_data[i][:-1])
    # Remove first character for target sequence
    target_seq.append(train_data[i][1:])

# create integer array for each sentence. each integer represent the character dictionary index
for i in range(len(train_data)):
    input_seq[i] = [word_to_index[character] for character in input_seq[i]]
    target_seq[i] = [word_to_index[character] for character in target_seq[i]]

# turn the input sequence to one-hot vectors array
input_seq = oneHotVector(input_seq, vocabulary_size, max_length - 1, len(train_data))

# turn numpy data to tensors
input_seq = torch.from_numpy(input_seq)
target_seq = torch.Tensor(target_seq)


"""" ------------- Main - data process and training -------------------- """

# create the RNN object
our_rnn = RNN(vocabulary_size, vocabulary_size, 64, 1)

# Define hyperparameters
epochs = 10000
lr = 0.005

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(our_rnn.parameters(), lr=lr)

# train the network
for epoch in range(epochs+1):
    # reset the gradient from previous epochs
    optimizer.zero_grad()
    # feed forward the input sequence
    output, hidden = our_rnn.forward(input_seq)
    # calculate the loss
    loss = criterion(output, target_seq.view(-1).long())
    # back propagate to calculate gradient
    loss.backward()
    # update weights
    optimizer.step()

    if epoch % 10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))


"""" ------------- Main - predictions-------------------- """

def predictNextItem(char):
    # turn character to input to our network
    char = np.array([[word_to_index[c] for c in char]])
    char = oneHotVector(char, vocabulary_size, char.shape[1], 1)
    char = torch.from_numpy(char)


    # process the input
    out, hidden = our_rnn.forward(char)

    # predict the result and return it
    prob = nn.functional.softmax(out[-1], dim=0).data

    char_ind = torch.max(prob, dim=0)[1].item()

    return index_to_word[char_ind], hidden



def generate(out_len, start):
    # turn to evaluation mode
    our_rnn.eval()

    # set empty chars array for results
    chars = []

    # First off, run through the starting characters
    chars = [ch for ch in start]
    size = out_len - len(chars)

    # Now pass in the previous characters and get a new one
    for k in range(size):
        char, h = predictNextItem(chars)
        chars.append(char)

    return ''.join(chars)


"""" ------------- Main - user interface -------------------- """

user_input = input("if you want to exit type 'ex' \n sentence to continue: ")

while True:
    if user_input == 'ex':
        print("exits...")
        break
    res = generate(30, user_input)
    print(res, '\n')
    user_input = input("sentence to continue: ")























