import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from module_train.basic_model.model_architecture_dl.sub_layer.cnn_feature_extract import CNNFeatureExtract1D, CNNFeatureExtract2D
from module_train.basic_model.model_architecture_dl.sub_layer.layer_high_way import Highway
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def xavier_uniform_init(m):
    """
    Xavier initializer to be used with model.apply
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight.data)


class LSTMCNNWord(nn.Module):
    def __init__(self, cf, vocabs):

        super(LSTMCNNWord, self).__init__()
        vocab_word, vocab_char, vocab_label = vocabs
        self.vocabs = vocabs
        self.cf = cf

        self.output_size = len(vocab_label)
        self.word_embedding_dim = cf['word_`embedding_dim']
        self.hidden_size_word = cf['hidden_size_word']

        self.char_embedding_dim = cf['char_embedding_dim']
        self.hidden_size_char_lstm = cf['hidden_size_char_lstm']

        self.word_embedding_layer = nn.Embedding(len(vocab_word), self.word_embedding_dim)
        self.char_embedding_layer = None

        self.use_highway_char = cf['use_highway_char']
        self.dropout_rate = cf['dropout_rate']

        self.use_char_cnn = cf['use_char_cnn']
        if self.use_char_cnn:
            self.char_cnn_filter_num = cf['char_cnn_filter_num']
            self.char_window_size = cf['char_window_size']
            self.dropout_cnn_char = cf['dropout_cnn_char']

        # this option for choose last layer :
        # can be: cnn (with convolution + max pooling)
        # can be max pooling
        # can be mean pooling
        self.option_last_layer = cf['option_last_layer']
        if self.option_last_layer == "cnn":
            self.cnn_filter_num = cf['cnn_filter_num']
            self.window_size = cf['window_size']
            self.dropout_cnn_word = cf['dropout_cnn_word']

        self.hidden_size_char = 0
        if vocab_char is not None and self.char_embedding_dim > 0:
            self.char_embedding_layer = nn.Embedding(len(vocab_char), self.char_embedding_dim)

            if not self.use_char_cnn:
                self.lstm_char = nn.LSTM(self.char_embedding_dim,
                                         self.hidden_size_char_lstm,
                                         num_layers=1,
                                         batch_first=True,
                                         bidirectional=False,
                                         dropout=self.dropout_rate)
                self.hidden_size_char = self.hidden_size_char_lstm
            else:
                if cf['D_cnn'] == '1_D':
                    self.layer_char_cnn = CNNFeatureExtract1D(self.char_embedding_dim,
                                                              self.char_cnn_filter_num,
                                                              self.char_window_size,
                                                              self.dropout_cnn_char)
                else:
                    self.layer_char_cnn = CNNFeatureExtract2D(self.char_embedding_dim,
                                                              self.char_cnn_filter_num,
                                                              self.char_window_size,
                                                              self.dropout_cnn_char)

                self.hidden_size_char = self.char_cnn_filter_num * len(self.char_window_size)

            if self.use_highway_char:
                self.highway_char = Highway(self.hidden_size_char, num_layers=1, f=torch.relu)

        self.embedding_word_lstm = self.word_embedding_dim + self.hidden_size_char
        if vocab_word.vectors is not None:
            if self.word_embedding_dim != vocab_word.vectors.shape[1]:
                raise ValueError("expect embedding word: {} but got {}".format(self.word_embedding_dim,
                                                                               vocab_word.vectors.shape[1]))

            self.word_embedding_layer.weight.data.copy_(vocab_word.vectors)
            self.word_embedding_layer.requires_grad = False

        self.lstm_word = nn.LSTM(self.embedding_word_lstm,
                                 self.hidden_size_word,
                                 num_layers=1,
                                 batch_first=True,
                                 bidirectional=True,
                                 dropout=self.dropout_rate)

        if self.option_last_layer == "cnn":
            if cf['D_cnn'] == '1_D':
                self.layer_cnn_for_lstm = CNNFeatureExtract1D(self.hidden_size_word * 2,
                                                              self.cnn_filter_num,
                                                              self.window_size,
                                                              self.dropout_cnn_word)
            else:
                self.layer_cnn_for_lstm = CNNFeatureExtract2D(self.hidden_size_word * 2,
                                                              self.cnn_filter_num,
                                                              self.window_size,
                                                              self.dropout_cnn_word)

            self.hidden_final = self.cnn_filter_num * len(self.window_size)
        else:
            self.hidden_final = self.hidden_size_word * 2
        self.dropout = nn.Dropout(self.dropout_rate)
        self.label = nn.Linear(self.hidden_final, self.output_size)

    def compute(self, batch, training=True):
        # input_word_emb = [batch_size, seq_sent, word_emb_dim]
        inputs_word_emb = self.word_embedding_layer(batch.inputs_word[0])
        inputs_word_emb = inputs_word_emb.permute(0, 2, 1)
        inputs_word_emb = F.dropout2d(inputs_word_emb, self.dropout_rate, training=training)
        inputs_word_emb = inputs_word_emb.permute(0, 2, 1)

        if self.char_embedding_layer is not None:

            # batch.char = [batch_size, seq_len, max_len_word]
            # input_char_emb = [batch x seq_len, max_len_word, char_emb_dim]
            inputs_char_emb = self.char_embedding_layer(batch.inputs_char.view(-1, batch.inputs_char.shape[-1]))
            inputs_char_emb = self.dropout(inputs_char_emb)

            if not self.use_char_cnn:
                seq_len = inputs_word_emb.shape[1]

                # final_hidden_state_char = [1, batch x seq_len, hidden_size_char]
                _, (final_hidden_state_char, _) = self.lstm_char(inputs_char_emb)

                # input_char_emb = [batch, seq_len, hidden_size_char]
                output_char_layer = final_hidden_state_char.view(-1, seq_len, self.hidden_size_char)
            else:
                output_char_conv_layer = self.layer_char_cnn(inputs_char_emb)
                output_char_layer = output_char_conv_layer.view(batch.inputs_char.shape[0],
                                                                batch.inputs_char.shape[1],
                                                                output_char_conv_layer.shape[1])

            if self.use_highway_char:
                output_char_layer = self.highway_char(output_char_layer)

            inputs_word_emb = torch.cat([inputs_word_emb, output_char_layer], -1)

        # need packed sequence
        packed_inputs_word_emb = pack_padded_sequence(inputs_word_emb, batch.inputs_word[1], batch_first=True)
        pack_output_hidden_word, (_, _) = self.lstm_word(packed_inputs_word_emb)

        # unpack sequence
        output_hidden_word, _ = pad_packed_sequence(pack_output_hidden_word, batch_first=True)

        # after output we do cnn for lstm hidden step for classify or just max pooling or mean pooling
        if self.option_last_layer == "cnn":
            final_output = self.layer_cnn_for_lstm(output_hidden_word)
        elif self.option_last_layer == "max_pooling":
            final_output = torch.max(output_hidden_word, 1)[0]
        else:
            # mean pooling last layer
            final_output = torch.mean((output_hidden_word, 1))
        return final_output

    def forward(self, batch, training=False):
        with torch.no_grad():
            final_output = self.compute(batch, training)
            logits = self.label(final_output)
        return logits

    def loss(self, batch):
        target = batch.labels
        final_output = self.compute(batch)

        logits = self.label(final_output)
        # class_weights = torch.FloatTensor([0.2, 0.42, 0.38]).cuda()
        loss = F.cross_entropy(logits, target)

        predict_value = torch.max(logits, 1)[1]
        list_predict = predict_value.cpu().numpy().tolist()
        list_target = target.cpu().numpy().tolist()

        return loss, list_predict, list_target

    @classmethod
    def create(cls, path_folder_model, cf, vocabs, device_set="cuda:0"):
        model = cls(cf, vocabs)
        if cf['use_xavier_weight_init']:
            model.apply(xavier_uniform_init)

        if torch.cuda.is_available():
            device = torch.device(device_set)
            model = model.to(device)

        path_vocab_file = os.path.join(path_folder_model, "vocabs.pt")
        torch.save(vocabs, path_vocab_file)

        path_config_file = os.path.join(path_folder_model, "model_cf.json")
        with open(path_config_file, "w") as w_config:
            json.dump(cf, w_config)

        return model

    @classmethod
    def load(cls, path_folder_model, path_model_checkpoint):
        path_vocab_file = os.path.join(path_folder_model, 'vocabs.pt')
        path_config_file = os.path.join(path_folder_model, 'model_cf.json')

        if not os.path.exists(path_vocab_file) or \
                not os.path.exists(path_config_file) or \
                not os.path.exists(path_model_checkpoint):
            raise OSError(" 1 of 3 file does not exist")

        vocabs = torch.load(path_vocab_file)
        with open(path_config_file, "r") as r_config:
            cf = json.load(r_config)

        model = cls(cf, vocabs)
        if torch.cuda.is_available():
            model = model.cuda()
            model.load_state_dict(torch.load(path_model_checkpoint))
        else:
            model.load_state_dict(torch.load(path_model_checkpoint, map_location=lambda storage, loc: storage))
        return model

    def save(self, path_save_model, name_model):
        checkpoint_path = os.path.join(path_save_model, name_model)
        torch.save(self.state_dict(), checkpoint_path)
