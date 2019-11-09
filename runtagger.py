# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class LSTMTagger(nn.Module):
    def __init__(self, num_layers, batch_size, device, word_to_index, tag_to_index, num_lstm_units=44, embedding_dim=32):
        super(LSTMTagger, self).__init__()
        self.to(device)
        self.on_gpu = True

        self.vocabulary = word_to_index
        self.tags = tag_to_index

        self.num_layers = num_layers
        self.num_lstm_units = num_lstm_units
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.device = device

        self.num_tags = len(self.tags)

        self.__build_model()

    def __build_model(self):
        vocabulary_size = len(self.vocabulary)
        padding_idx = self.vocabulary['<PADDING>']

        # Word embedding
        self.word_embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=self.embedding_dim,
            padding_idx=padding_idx
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.num_lstm_units,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional = True, 
            dropout = 0.5, # TODO:
        )

        # Output layer
        self.hidden_to_tag = nn.Linear(
            self.num_lstm_units * 2,
            self.num_tags
        )
    
    def init_hidden(self):
        # weight = next(self.parameters()).data
        # hidden = (weight.new(self.num_layers * 2, self.batch_size, self.num_lstm_units).zero_().to(self.device),
        #               weight.new(self.num_layers * 2, self.batch_size, self.num_lstm_units).zero_().to(self.device))
        # return hidden 

        hidden_1 = torch.randn(self.num_layers * 2, self.batch_size, self.num_lstm_units)
        hidden_2 = torch.randn(self.num_layers * 2, self.batch_size, self.num_lstm_units)

        if self.on_gpu:
            hidden_1 = hidden_1.cuda()
            hidden_2 = hidden_2.cuda()

        hidden_1 = Variable(hidden_1)
        hidden_2 = Variable(hidden_2)

        return (hidden_1, hidden_2)

    def forward(self, x, x_lengths):
        # Reset the LSTM hidden state.
        self.hidden = self.init_hidden()
        batch_size, seq_len = x.size()

        # Input transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        x = self.word_embedding(x)

        # Hidden state: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, num_lstm_units)
        x_lengths = torch.tensor([x_lengths]* self.batch_size, dtype=torch.long).to(self.device)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True) # Don't show padded items to the model

        x, self.hidden = self.lstm(x, self.hidden)

        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # Tag probabilities: (batch_size, seq_len, num_lstm_units) -> (batch_size * seq_len, num_lstm_units)
        x = x.contiguous()
        x = x.view(-1, x.shape[2])
        x = self.hidden_to_tag(x)

        # Softmax: (batch_size * seq_len, num_lstm_units) -> (batch_size, seq_len, num_tags)
        x = F.log_softmax(x, dim=1)
        x = x.view(batch_size, seq_len, self.num_tags)
        predicted_labels = x

        return predicted_labels

    def loss(self, predicted_labels, actual_labels):
        actual_labels = actual_labels.view(-1)
        predicted_labels = predicted_labels.view(-1, self.num_tags)

        # Filter all tokens that aren't <PADDING>
        tag_pad_token = self.tags['<PADDING>']
        mask = (actual_labels > tag_pad_token).float()

        num_tokens = int(torch.sum(mask).data)
        predicted_labels = predicted_labels[range(predicted_labels.shape[0]), actual_labels] * mask
        cross_entropy_loss = -torch.sum(predicted_labels) / num_tokens

        return cross_entropy_loss


def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.
    # use torch library to load model_file

    startTime = datetime.datetime.now()
        
    test_lines = []
    with open(test_file) as t:
        for line in t:
            test_lines.append(line)
   
    max_length, b, word_to_index, tag_to_index, model_state_dict = torch.load(model_file)
    index_to_tag = {v:k for k,v in tag_to_index.items()}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = LSTMTagger(2, b, device, word_to_index, tag_to_index)
    model.load_state_dict(model_state_dict)
    model.to(device)

    with open(out_file, 'w') as out_writer:
        for line in test_lines:
            line = line.split()

            # We don't train here since we are evaluating. So, the code is wrapped in torch.no_grad()
            with torch.no_grad():
                inputs = prepare_sequence([line] * b, word_to_index, max_length, device)
                tag_scores = model(inputs, max_length)

                v, tag_indices = torch.max(tag_scores[0], 1)
                tag_names = [index_to_tag[idx.item()] for idx in tag_indices]

            output = ""
            for word, tag in zip(line, tag_names):
                output += word + "/" + tag + " "

            out_writer.write(output + "\n")


    print("Finished in %s seconds" % (datetime.datetime.now() - startTime))
    # print('Finished...')


def prepare_sequence(lines, index_dict, max_length, device):
    new_lines = []

    for line in lines:
        indices = []
        
        for word in line:
            if word in index_dict:
                indices.append(index_dict[word])
            else:
                indices.append(index_dict['<UNKNOWN>'])

        indices += [index_dict['<PADDING>']] * (max_length - len(indices))
        new_lines.append(indices)

    return torch.tensor(new_lines, dtype=torch.long).to(device)


if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)
