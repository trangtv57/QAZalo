import torch
import torch.nn as nn
from module_train.basic_model.model_architecture_dl.sub_layer.cnn_feature_extract import CNNFeatureExtract1D
from module_train.basic_model.model_architecture_dl.sub_layer.layer_high_way import Highway
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class LSTMWordCnnCharEncoder(nn.Module):
    def __init__(self,
                 vocabs_word,
                 word_embedding_dim,
                 hidden_size_word,
                 vocabs_char=None,
                 char_embedding_dim=0,
                 hidden_size_char=0,
                 use_char_cnn=False,
                 use_highway=False,
                 char_cnn_filter_num=0,
                 char_window_size=[0],
                 dropout_cnn_char=0.1,
                 dropout_rate=0.5,
                 option_last_layer="max_pooling"):

        super(LSTMWordCnnCharEncoder, self).__init__()
        self.word_embedding_layer = nn.Embedding(len(vocabs_word), word_embedding_dim)

        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.hidden_size_word = hidden_size_word
        self.dropout_rate = dropout_rate
        self.use_char_cnn = use_char_cnn
        self.use_highway = use_highway
        # this option for choose last layer :
        # can be: cnn (with convolution + max pooling)
        # can be max pooling
        # can be mean pooling
        self.option_last_layer = option_last_layer
        if self.use_char_cnn:
            self.char_cnn_filter_num = char_cnn_filter_num
            # print(self.char_cnn_filter_num)
            self.char_window_size = char_window_size
            self.dropout_cnn_char = dropout_cnn_char

        if vocabs_word.vectors is not None:
            if self.word_embedding_dim != vocabs_word.vectors.shape[1]:
                raise ValueError("expect embedding word: {} but got {}".format(self.word_embedding_dim,
                                                                               vocabs_word.vectors.shape[1]))

            self.word_embedding_layer.weight.data.copy_(vocabs_word.vectors)
            self.word_embedding_layer.requires_grad = False

        # setting for encoder query
        self.hidden_size_char = 0
        if self.char_embedding_dim > 0:
            self.char_embedding_layer = nn.Embedding(len(vocabs_char), self.char_embedding_dim)
            if not use_char_cnn:
                self.lstm_char = nn.LSTM(char_embedding_dim,
                                         hidden_size_char,
                                         num_layers=1,
                                         batch_first=True,
                                         bidirectional=False,
                                         dropout=self.dropout_rate)

                self.hidden_size_char = hidden_size_char
            else:
                self.layer_char_cnn = nn.ModuleList([CNNFeatureExtract1D(self.char_embedding_dim,
                                                          self.char_window_size,
                                                          self.dropout_cnn_char)
                                                    for i in range(self.char_cnn_filter_num)])

                self.hidden_size_char = self.char_cnn_filter_num * len(self.char_window_size) *\
                                            self.char_embedding_dim

        self.word_embedding_dim += self.hidden_size_char
        if self.use_highway:
            self.highway_layer = Highway(self.word_embedding_dim, num_layers=2, f=torch.relu)

        self.lstm_word = nn.LSTM(self.word_embedding_dim,
                                 self.hidden_size_word,
                                 num_layers=1,
                                 batch_first=True,
                                 bidirectional=True,
                                 dropout=self.dropout_rate)

        if self.option_last_layer == "cnn":
            self.layer_cnn_for_lstm = CNNFeatureExtract1D(self.hidden_size_word * 2,
                                                          self.window_size,
                                                          self.dropout_cnn)
            self.hidden_final = self.cnn_filter_num * len(self.window_size)
        else:
            self.hidden_final = self.hidden_size_word * 2
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, batch_word, batch_char):
        # input_word_emb = [batch_size, seq_sent, word_emb_dim]
        inputs_word_emb = self.word_embedding_layer(batch_word[0])
        inputs_word_emb = self.dropout(inputs_word_emb)
        seq_len_word = batch_word[1]
        # get index sort for descending (using for packed)
        index_sort_first = torch.argsort(seq_len_word, dim=0, descending=True)
        seq_len_word_sort = seq_len_word.index_select(0, index_sort_first)
        index_get_origin = torch.argsort(index_sort_first, dim=0, descending=False)

        if self.char_embedding_dim > 0:
            # batch.char = [batch_size, seq_len, max_len_word]
            # input_char_emb = [batch x seq_len, max_len_word, char_emb_dim]
            inputs_char_emb = self.char_embedding_layer(batch_char.view(-1, batch_char.shape[-1]))
            inputs_char_emb = self.dropout(inputs_char_emb)

            if not self.use_char_cnn:
                seq_len = inputs_word_emb.shape[1]
                # final_hidden_state_char = [1, batch x seq_len, hidden_size_char]
                _, (final_hidden_state_char, _) = self.lstm_char(inputs_char_emb)
                # input_char_emb = [batch, seq_len, hidden_size_char]
                output_char_layer = final_hidden_state_char.view(-1, seq_len, self.hidden_size_char)
            else:
                output_char_conv_layer = self.layer_char_cnn[0](inputs_char_emb)

                for i in range(1, self.char_cnn_filter_num):
                    out_each_cnn_layer_char = self.layer_char_cnn[i](inputs_char_emb)
                    output_char_conv_layer = torch.cat([output_char_conv_layer, out_each_cnn_layer_char], -1)

                output_char_layer = output_char_conv_layer.view(batch_char.shape[0],
                                                                batch_char.shape[1],
                                                                output_char_conv_layer.shape[1])

            inputs_word_emb = torch.cat([inputs_word_emb, output_char_layer], -1)

        if self.use_highway:
            inputs_word_emb = self.dropout(self.highway_layer(inputs_word_emb))

        # test pack and unpack
        inputs_word_emb_sort = inputs_word_emb.index_select(0, index_sort_first)
        pack_input_word_emb = pack_padded_sequence(inputs_word_emb_sort, seq_len_word_sort, batch_first=True)
        pack_output_hidden_word, (_, _) = self.lstm_word(pack_input_word_emb)

        # unpack with index get origin
        output_hidden_word, _ = pad_packed_sequence(pack_output_hidden_word, batch_first=True)
        output_hidden_word = output_hidden_word.index_select(0, index_get_origin)

        output_hidden_word = self.dropout(output_hidden_word)

        # after output we do cnn for lstm hidden step for classify or just max pooling or mean pooling
        if self.option_last_layer == "cnn":
            final_output = self.layer_cnn_for_lstm(output_hidden_word)

        elif self.option_last_layer == "max_pooling":
            final_output = torch.max(output_hidden_word, 1)[0]

        elif self.option_last_layer == "mean_pooling":

            final_output = torch.mean((output_hidden_word, 1))

        elif self.option_last_layer == "origin":
            final_output = output_hidden_word

        return final_output
