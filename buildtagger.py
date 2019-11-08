# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class CharacterEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, filter_size, hidden_dim, pad_idx):
        super().__init__()
        
        assert filter_size % 2 == 1, "Kernel size must be odd!"
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.cnn = nn.Conv1d(in_channels = embedding_dim,
                             out_channels = hidden_dim,
                             kernel_size = filter_size,
                             padding = (filter_size - 1) // 2)
        
    def forward(self, chars):
        batch_size = chars.shape[0]
        sent_len = chars.shape[1]
        word_len = chars.shape[2]
        
        embedded = self.embedding(chars)
        
        embedded = embedded.view(-1, word_len, self.embedding_dim)
        embedded = embedded.permute(0, 2, 1)
        
        embedded = self.cnn(embedded)
                
        embedded = embedded.view(sent_len, batch_size, self.hidden_dim, word_len)
                
        embedded = torch.max(embedded, dim = -1).values
        embedded = embedded.permute(1, 0, 2)
        
        return embedded

class LSTMTagger(nn.Module):
    def __init__(self, num_layers, batch_size, device, word_to_index, char_to_index, tag_to_index, num_lstm_units=50, embedding_dim=32):
        super(LSTMTagger, self).__init__()
        self.to(device)
        self.on_gpu = True

        self.vocabulary = word_to_index
        self.char_to_index = char_to_index
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
            padding_idx=padding_idx,
        )

        padding_idx = self.char_to_index['<PADDING>'] 
        self.char_encoder = CharacterEncoder(
            vocab_size=len(self.char_to_index),
            embedding_dim=50, # TODO:
            filter_size=3, 
            hidden_dim=self.embedding_dim, 
            pad_idx=padding_idx,
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
            self.num_tags,
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

    def forward(self, x, x_lengths, chars):
        # Reset the LSTM hidden state for each sequence
        self.hidden = self.init_hidden()
        batch_size, seq_len = x.size()

        # Input transformation: 
        #   (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        #   (batch_size, seq_len, word_len) -> (batch_size, seq_len, embedding_dim)
        x = self.word_embedding(x)
        chars_embedded = self.char_encoder(chars)

        embedded = torch.cat((chars_embedded, x), dim = -1)

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

def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    # use torch library to save model parameters, hyperparameters, etc. to model_file
    startTime = datetime.datetime.now()

    train_lines = []
    train_tags = []
    with open(train_file) as infile:
        for line in infile:
            train_line, train_tag = get_line_data(line)
            train_lines.append(train_line)
            train_tags.append(train_tag)


    word_to_index_counts = {}
    word_to_index = {
        '<PADDING>': 0,
        '<UNKNOWN>': 1,
    }
    char_to_index_counts = {}
    char_to_index = {
        '<PADDING>': 0,
        '<UNKNOWN>': 1,
    }
    tag_to_index = {
        '<PADDING>': 0,
    }

    for line, tags in zip(train_lines, train_tags):
        for word, tag in zip(line, tags):
            if word in word_to_index_counts:
                word_to_index_counts[word][1] += 1
            else:
                word_to_index_counts[word] = [len(word_to_index_counts), 1] # Word IDs (index) and count
            
            for c in word:
                if c in char_to_index_counts:
                    char_to_index_counts[c][1] += 1
                else:
                    char_to_index_counts[c] = [len(char_to_index_counts), 1] # Char IDs (index) and count

            if tag not in tag_to_index:
                tag_to_index[tag] = len(tag_to_index) # Tag IDs (index)
            
    for word, index_count in word_to_index_counts.items():
        # Let words occurring 1 time be considered as <UNKNOWN> during training
        if index_count[1] > 1:
            word_to_index[word] = len(word_to_index)

    for char, index_count in char_to_index_counts.items():
        # Let characters occurring 1 time be considered as <UNKNOWN> during training
        if index_count[1] > 1:
            char_to_index[char] = len(char_to_index)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_length = 150
    max_word_length = 75
    b = 32 # TODO: Batch size (32?)

    model = LSTMTagger(2, b, device, word_to_index, char_to_index, tag_to_index)
    model.to(device)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay = 0.0005)
    
    # for epoch in range(7): # FIXME: 10?
    for epoch in range(3):
        print('epoch', epoch)

        if epoch == 1:
            for group in optimizer.param_groups:
                group['lr'] = 0.05
        elif epoch == 2:
            for group in optimizer.param_groups:
                group['lr'] = 0.05

        for i in range(int(len(train_lines)/b)):
            model.zero_grad()
            
            start = i * b
            lines = train_lines[start:start + b]
            tags = train_tags[start:start + b]

            words = prepare_sequence(lines, word_to_index, max_length, device)
            chars = prepare_char_sequence(lines, char_to_index, max_length, max_word_length, device)
            target_labels = prepare_sequence(tags, tag_to_index, max_length, device)

            tag_scores = model(words, max_length, chars)

            loss = model.loss(tag_scores, target_labels)
            loss.backward()
            optimizer.step()


    torch.save((max_length, max_word_length, b, word_to_index, char_to_index, tag_to_index, model.state_dict()), model_file)

    print("Finished in %s seconds" % (datetime.datetime.now() - startTime))
    # print("Finished...")

"""
Returns the input sentences as tensors.
"""
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

def prepare_char_sequence(lines, index_dict, max_length, max_word_length, device):
    new_lines = []

    word_padding = []
    for i in range(max_word_length):
        word_padding.append(index_dict['<PADDING>'])

    for line in lines:
        indices = []
        
        for word in line:
            char_indices = []
            for c in word:
                if c in index_dict:
                    char_indices.append(index_dict[c])
                else:
                    char_indices.append(index_dict['<UNKNOWN>'])

            char_indices += [index_dict['<PADDING>']] * (max_word_length - len(char_indices))
            indices.append(char_indices)
            
        for i in range((max_length - len(indices))):
            indices.append(word_padding) 

        new_lines.append(indices)
    
    return torch.tensor(new_lines).to(device)
    # return new_lines

"""
Returns the sentence (line) as [words, tags].
"""
def get_line_data(line):
    words = []
    tags = []
    
    terms = line.split()
    for term in terms:
        splitIdx = term.rfind('/')
        word = term[:splitIdx]
        tag = term[splitIdx+1:]
        
        words.append(word)
        tags.append(tag)

    return (words, tags)


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
