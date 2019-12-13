import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from module_train.basic_model.model_architecture_dl.sub_layer import LSTMWordCnnCharEncoder


def xavier_uniform_init(m):
    """
    Xavier initializer to be used with model.apply
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight.data)


class LSTMCNNWordInferSent(nn.Module):
    def __init__(self, cf, vocabs, device=None):

        super(LSTMCNNWordInferSent, self).__init__()
        vocab_word, vocab_char, vocab_label = vocabs
        self.vocabs = vocabs
        self.cf = cf
        self.device = device

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

        self.hidden_layer_1_dim = cf['hidden_layer_1_dim']
        self.hidden_layer_2_dim = cf['hidden_layer_2_dim']
        self.weight_class = cf['weight_class']

        # this option for choose last layer :
        # can be: cnn (with convolution + max pooling)
        # can be max pooling
        # can be mean pooling
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
                                                    self.option_last_layer)

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
                                                       self.option_last_layer)

        if self.hidden_layer_1_dim != 0:
            self.hidden_layer_1 = nn.Linear(self.hidden_size_word * 8, self.hidden_layer_1_dim)
            if self.hidden_layer_2_dim != 0:
                self.hidden_layer_2 = nn.Linear(self.hidden_layer_1_dim, self.hidden_layer_2_dim)
                self.label = nn.Linear(self.hidden_layer_2_dim, len(vocab_label))

                self.classifier = nn.Sequential(*[self.hidden_layer_1,
                                                  nn.ReLU(),
                                                  nn.Dropout(self.dropout_rate),
                                                  self.hidden_layer_2,
                                                  nn.ReLU(),
                                                  nn.Dropout(self.dropout_rate),
                                                  self.label
                                                  ])

            else:
                self.label = nn.Linear(self.hidden_layer_1_dim, len(vocab_label))
                self.classifier = nn.Sequential(*[self.hidden_layer_1,
                                                  nn.ReLU(),
                                                  nn.Dropout(self.dropout_rate),
                                                  self.label
                                                  ])
        else:
            self.label = nn.Linear(self.hidden_size_word * 8, len(vocab_label))
            self.classifier = nn.Sequential(*[self.label])

    def compute_logits(self, batch):
        feature_document = self.encoder_document(batch.inputs_word_document, batch.inputs_char_document)
        feature_query = self.encoder_query(batch.inputs_word_query, batch.inputs_char_query)

        pair_feature = torch.cat([feature_document,
                                  feature_query,
                                  torch.abs(feature_document - feature_query),
                                  feature_document * feature_query], 1)
        logits = self.classifier(pair_feature)
        return logits

    def forward(self, batch):
        with torch.no_grad():
            logits = self.classifier(batch)
        return logits

    def loss(self, batch):
        target = batch.labels
        logits = self.compute_logits(batch)

        if self.device is not None:
            class_weights = torch.FloatTensor(self.weight_class).cuda(self.device)
        else:
            class_weights = torch.FloatTensor(self.weight_class)

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
