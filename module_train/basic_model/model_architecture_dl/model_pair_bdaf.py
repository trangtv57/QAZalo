# https://github.com/jojonki/BiDAF/blob/master/layers/bidaf.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from module_train.basic_model.model_architecture_dl.sub_layer import LSTMWordCnnCharEncoder
from module_train.basic_model.model_architecture_dl.sub_layer.cnn_feature_extract import CNNFeatureExtract1D


def xavier_uniform_init(m):
    """
    Xavier initializer to be used with model.apply
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight.data)


class BiDAF(nn.Module):
    def __init__(self,  cf, vocabs, device=None):
        super(BiDAF, self).__init__()

        vocab_word, vocab_char, vocab_label = vocabs
        self.vocabs = vocabs
        self.cf = cf
        self.device = device

        self.weight_class = cf['weight_class']

        self.output_size = len(vocab_label)
        self.word_embedding_dim = cf['word_embedding_dim']
        self.hidden_size_word = cf['hidden_size_word']

        self.char_embedding_dim = cf['char_embedding_dim']
        self.hidden_size_char_lstm = cf['hidden_size_char_lstm']
        self.use_char_cnn = cf['use_char_cnn']

        if self.use_char_cnn:
            self.char_cnn_filter_num = cf['char_cnn_filter_num']
            self.char_window_size = cf['char_window_size']
            self.dropout_cnn_char = cf['dropout_cnn_char']
        else:
            self.char_cnn_filter_num = 0
            self.char_window_size = [0]
            self.dropout_cnn_char = 0

        self.use_highway = cf['use_highway']
        self.dropout_rate = cf['dropout_rate']

        self.dropout_layer = nn.Dropout(self.dropout_rate)
        # this option for choose last layer :
        # can be: cnn (with convolution + max pooling)
        # can be max pooling
        # can be mean pooling
        self.use_modeling_in_last = cf['use_modeling_in_last']
        if self.use_modeling_in_last:
            self.num_layer_modeling_lstm = cf['num_layer_modeling_lstm']

        self.option_last_layer = cf['option_last_layer']
        if self.option_last_layer == "cnn":
            self.cnn_filter_num = cf['cnn_filter_num']
            self.window_size = cf['window_size']
            self.dropout_cnn = cf['dropout_cnn']

        # setting for encoder query
        self.encoder_query = LSTMWordCnnCharEncoder(vocab_word,
                                                    self.word_embedding_dim,
                                                    self.hidden_size_word,
                                                    vocab_char,
                                                    self.char_embedding_dim,
                                                    self.hidden_size_char_lstm,
                                                    self.use_char_cnn,
                                                    self.use_highway,
                                                    self.char_cnn_filter_num,
                                                    self.char_window_size,
                                                    self.dropout_cnn_char,
                                                    self.dropout_rate,
                                                    option_last_layer="origin")

        self.encoder_document = LSTMWordCnnCharEncoder(vocab_word,
                                                       self.word_embedding_dim,
                                                       self.hidden_size_word,
                                                       vocab_char,
                                                       self.char_embedding_dim,
                                                       self.hidden_size_char_lstm,
                                                       self.use_char_cnn,
                                                       self.use_highway,
                                                       self.char_cnn_filter_num,
                                                       self.char_window_size,
                                                       self.dropout_cnn_char,
                                                       self.dropout_rate,
                                                       option_last_layer="origin")

        self.W = nn.Linear(6 * self.hidden_size_word, 1, bias=False)

        self.last_dim_layer = self.hidden_size_word * 8
        if self.use_modeling_in_last:
            self.modeling_layer = nn.GRU(self.last_dim_layer,
                                         self.hidden_size_word,
                                         num_layers=self.num_layer_modeling_lstm,
                                         bidirectional=True,
                                         dropout=self.dropout_rate,
                                         batch_first=True)
            self.last_dim_layer = self.hidden_size_word * 10

        if self.option_last_layer == "cnn":
            self.layer_cnn_for_lstm = nn.ModuleList([CNNFeatureExtract1D(self.last_dim_layer,
                                                                         self.window_size,
                                                                         self.dropout_cnn)
                                                     for i in range(self.cnn_filter_num)])
            self.hidden_final = self.cnn_filter_num * len(self.window_size) * self.last_dim_layer
        else:
            self.hidden_final = self.last_dim_layer

        self.label = nn.Linear(self.hidden_final, len(vocab_label))

    def compute(self, batch):
        # 1. Character Embedding Layer
        # 2. Word Embedding Layer
        # 3. Contextual  Embedding Layer
        batch_size = batch.inputs_word_document[0].size(0)
        context_length = batch.inputs_word_document[0].size(1)  # context sentence length (word level)
        query_length = batch.inputs_word_query[0].size(1)  # query sentence length   (word level)

        feature_document = self.encoder_document(batch.inputs_word_document, batch.inputs_char_document)
        feature_query = self.encoder_query(batch.inputs_word_query, batch.inputs_char_query)

        # 4. Attention Flow Layer
        # Make a similarity matrix
        shape = (batch_size, context_length, query_length, 2 * self.hidden_size_word)  # (N, T, J, 2d)
        embd_context_ex = feature_document.unsqueeze(2)  # (N, T, 1, 2d)
        embd_context_ex = embd_context_ex.expand(shape)  # (N, T, J, 2d)
        embd_query_ex = feature_query.unsqueeze(1)  # (N, 1, J, 2d)
        embd_query_ex = embd_query_ex.expand(shape)  # (N, T, J, 2d)

        a_elmwise_mul_b = torch.mul(embd_context_ex, embd_query_ex)  # (N, T, J, 2d)
        cat_data = torch.cat((embd_context_ex, embd_query_ex, a_elmwise_mul_b), 3)  # (N, T, J, 6d), [h;u;h◦u]
        S = self.W(cat_data).view(batch_size, context_length, query_length)  # (N, T, J)
        S = self.dropout_layer(S)

        # Context2Query
        c2q = torch.bmm(F.softmax(S, dim=-1), feature_query)  # (N, T, 2d) = bmm( (N, T, J), (N, J, 2d) )
        # Query2Context
        # b: attention weights on the context
        b = F.softmax(torch.max(S, 2)[0], dim=-1)  # (N, T)
        q2c = torch.bmm(b.unsqueeze(1), feature_document)  # (N, 1, 2d) = bmm( (N, 1, T), (N, T, 2d) )
        q2c = q2c.repeat(1, context_length, 1)  # (N, T, 2d), tiled T times

        # G: query aware representation of each context word
        # it's output attention
        output_bidaf = torch.cat((feature_document, c2q, feature_document.mul(c2q), feature_document.mul(q2c)), 2)  # (N, T, 8d)

        output_modeling = output_bidaf
        # check option for last layer of bdaf
        if self.use_modeling_in_last:
            output_modeling, _ = self.modeling_layer(output_bidaf)
            output_modeling = torch.cat((output_modeling, output_bidaf), 2)

        output_modeling = self.dropout_layer(output_modeling)

        if self.option_last_layer == "cnn":
            final_output = self.layer_cnn_for_lstm[0](output_modeling)
            for i in range(1, self.cnn_filter_num):
                output_each_cnn_layer = self.layer_cnn_for_lstm(output_modeling)
                final_output = torch.cat([final_output, output_each_cnn_layer], -1)

        elif self.option_last_layer == "max_pooling":
            final_output = torch.max(output_modeling, 1)[0]

        elif self.option_last_layer == "mean_pooling":
            final_output = torch.mean((output_modeling, 1))

        else:
            # origin: just use last time step for predict:
            final_output = output_modeling[-1]

        return final_output

    def forward(self, batch):
        with torch.no_grad():
            final_output = self.compute(batch)
            logits = self.label(final_output)
        return logits

    def loss(self, batch):
        target = batch.labels
        final_output = self.compute(batch)

        logits = self.label(final_output)
        if self.device is not None:
            class_weights = torch.FloatTensor(self.weight_class).cuda(self.device)
        else:
            class_weights = torch.FloatTensor([self.weight_class])

        loss = F.cross_entropy(logits, target, weight=class_weights)

        predict_value = torch.max(logits, 1)[1]
        list_predict = predict_value.cpu().numpy().tolist()
        list_target = target.cpu().numpy().tolist()

        return loss, list_predict, list_target

    @classmethod
    def create(cls, path_folder_model, cf, vocabs, device_set="cuda:0"):
        if torch.cuda.is_available():
            device = torch.device(device_set)
            model = cls(cf, vocabs, device)
            model = model.to(device)
        else:
            model = cls(cf, vocabs)

        if cf['use_xavier_weight_init']:
            model.apply(xavier_uniform_init)

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
