import os
import torch
from torchtext import data
from torchtext.vocab import Vectors
import logging
logger = logging.getLogger(__name__)


class MyPretrainedVector(Vectors):
    def __init__(self, name_file, cache):
        super(MyPretrainedVector, self).__init__(name_file, cache=cache)


def load_data_word_lstm_char(path_file_data,
                             name_file_train,
                             name_file_test=None,
                             device_set="cuda:0",
                             min_freq_word=1,
                             min_freq_char=1,
                             batch_size=2,
                             cache_folder=None,
                             name_vocab=None,
                             path_vocab_pre_built=None,
                             sort_key=True):

    inputs_word = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True, include_lengths=True)

    inputs_char_nesting = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>", batch_first=True)

    inputs_char = data.NestedField(inputs_char_nesting,
                                   init_token="<bos>", eos_token="<eos>")

    labels = data.LabelField(sequential=False)

    fields = ([(('inputs_word', 'inputs_char'), (inputs_word, inputs_char)), ('labels', labels)])

    if name_file_test is not None:
        train, test = data.TabularDataset.splits(path=path_file_data,
                                                 train=name_file_train,
                                                 test=name_file_test,
                                                 fields=tuple(fields),
                                                 format='csv',
                                                 skip_header=False,
                                                 csv_reader_params={'delimiter': '\t',
                                                                     'quoting': 3})

        if path_vocab_pre_built is None:
            if cache_folder is not None and name_vocab is not None:
                inputs_word.build_vocab(train.inputs_word, test.inputs_word, min_freq=min_freq_word,
                                        vectors=[MyPretrainedVector(name_vocab, cache_folder)])
            else:
                inputs_word.build_vocab(train.inputs_word, test.inputs_word, min_freq=min_freq_word)

            inputs_char.build_vocab(train.inputs_char, test.inputs_char, min_freq=min_freq_char)
            labels.build_vocab(train.labels)
        else:
            vocabs = torch.load(path_vocab_pre_built)
            inputs_word.vocab = vocabs[0]
            inputs_char.vocab = inputs_char_nesting.vocab = vocabs[1]
            labels.vocab = vocabs[2]

        if sort_key:
            train_iter, test_iter = data.BucketIterator.splits(datasets=(train, test),
                                                               batch_size=batch_size,
                                                               sort_key=lambda x: len(x.inputs_word),
                                                               sort_within_batch=True,
                                                               device=torch.device(device_set
                                                                                   if torch.cuda.is_available() else "cpu"))
        else:
            train_iter, test_iter = data.BucketIterator.splits(datasets=(train, test),
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               sort=False,
                                                               device=torch.device(device_set
                                                                                   if torch.cuda.is_available() else "cpu"))
        dict_return = {'iters': (train_iter, test_iter),
                       'vocabs': (inputs_word.vocab, inputs_char.vocab, labels.vocab)}
    else:
        path_file_data_train = path_file_data + name_file_train
        train = data.TabularDataset(path_file_data_train,
                                    fields=tuple(fields),
                                    format='csv',
                                    skip_header=False,
                                    csv_reader_params={'delimiter': '\t',
                                                       'quoting': 3})

        if path_vocab_pre_built is None:
            if cache_folder is not None and name_vocab is not None:
                inputs_word.build_vocab(train.inputs_word, min_freq=min_freq_word,
                                        vectors=[MyPretrainedVector(name_vocab, cache_folder)])
            else:
                inputs_word.build_vocab(train.inputs_word, min_freq=min_freq_word)

            inputs_char.build_vocab(train.inputs_char, min_freq=min_freq_char)
            labels.build_vocab(train.labels)
        else:
            vocabs = torch.load(path_vocab_pre_built)
            inputs_word.vocab = vocabs[0]
            inputs_char.vocab = inputs_char_nesting.vocab = vocabs[1]
            labels.vocab = vocabs[2]

        if sort_key:
            train_iter = data.BucketIterator(train,
                                             batch_size=batch_size,
                                             sort_key=lambda x: len(x.inputs_word),
                                             sort_within_batch=True,
                                             device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        else:
            train_iter = data.BucketIterator(train,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             sort=False,
                                             device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        dict_return = {'iters': [train_iter],
                       'vocabs': (inputs_word.vocab, inputs_char.vocab, labels.vocab)}

    return dict_return


def load_data_pair_task(path_file_data,
                        name_file_train,
                        name_file_test=None,
                        device_set="cuda:0",
                        min_freq_word=1,
                        min_freq_char=1,
                        batch_size=2,
                        cache_folder=None,
                        name_vocab=None,
                        path_vocab_pre_built=None):

    inputs_word_query = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True, include_lengths=True)
    inputs_char_query_nesting = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>", batch_first=True)
    inputs_char_query = data.NestedField(inputs_char_query_nesting, init_token="<bos>", eos_token="<eos>")

    inputs_word_document = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True, include_lengths=True)
    inputs_char_document_nesting = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>", batch_first=True)
    inputs_char_document = data.NestedField(inputs_char_document_nesting, init_token="<bos>", eos_token="<eos>")

    labels = data.LabelField(sequential=False)

    fields = ([(('inputs_word_query', 'inputs_char_query'), (inputs_word_query, inputs_char_query)),
               (('inputs_word_document', 'inputs_char_document'), (inputs_word_document, inputs_char_document)),
               ('labels', labels)])

    if name_file_test is not None:
        train, test = data.TabularDataset.splits(path=path_file_data,
                                                 train=name_file_train,
                                                 test=name_file_test,
                                                 fields=tuple(fields),
                                                 format='csv',
                                                 skip_header=False,
                                                 csv_reader_params={'delimiter': '\t',
                                                                    'quoting': 3})

        if path_vocab_pre_built is None:
            if cache_folder is not None and name_vocab is not None:
                inputs_word_document.build_vocab(train.inputs_word_document,
                                                 test.inputs_word_document,
                                                 min_freq=min_freq_word,
                                                 vectors=[MyPretrainedVector(name_vocab, cache_folder)])
            else:
                inputs_word_document.build_vocab(train.inputs_word_document,
                                                 test.inputs_word_document,
                                                 min_freq=min_freq_word)

            inputs_char_document.build_vocab(train.inputs_char_document, test.inputs_char_document,
                                             min_freq=min_freq_char)

            inputs_word_query.vocab = inputs_word_document.vocab
            inputs_char_query.vocab = inputs_char_query_nesting.vocab = \
                inputs_char_document_nesting.vocab = inputs_char_document.vocab
            labels.build_vocab(train.labels)
        else:
            vocabs = torch.load(path_vocab_pre_built)
            inputs_word_document.vocab = inputs_word_query.vocab = vocabs[0]
            inputs_char_document.vocab = inputs_char_query.vocab = \
                inputs_char_document_nesting.vocab = inputs_char_query_nesting.vocab = vocabs[1]
            labels.vocab = vocabs[2]

        train_iter, test_iter = data.BucketIterator.splits(datasets=(train, test),
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           sort=False,
                                                           device=torch.device(device_set
                                                                               if torch.cuda.is_available() else "cpu"))
        dict_return = {'iters': (train_iter, test_iter),
                       'vocabs': (inputs_word_document.vocab, inputs_char_document.vocab, labels.vocab)}
    else:
        path_file_data_train = os.path.join(path_file_data, name_file_train)
        train = data.TabularDataset(path_file_data_train,
                                    fields=tuple(fields),
                                    format='csv',
                                    skip_header=True,
                                    csv_reader_params={'delimiter': '\t',
                                                       'quoting': 3})

        if path_vocab_pre_built is None:
            if cache_folder is not None and name_vocab is not None:
                inputs_word_document.build_vocab(train.inputs_word_document, min_freq=min_freq_word,
                                        vectors=[MyPretrainedVector(name_vocab, cache_folder)])
            else:
                inputs_word_document.build_vocab(train.inputs_word_document, min_freq=min_freq_word)

            inputs_char_document.build_vocab(train.inputs_char_document, min_freq=min_freq_char)

            inputs_word_query.vocab = inputs_word_document.vocab
            inputs_char_query.vocab = inputs_char_query_nesting.vocab = \
                inputs_char_document_nesting.vocab = inputs_char_document.vocab

            labels.build_vocab(train.labels)

        else:
            vocabs = torch.load(path_vocab_pre_built)
            inputs_word_document.vocab = inputs_word_query.vocab = vocabs[0]
            inputs_char_document.vocab = inputs_char_query.vocab = \
                inputs_char_document_nesting.vocab = inputs_char_query_nesting.vocab = vocabs[1]
            labels.vocab = vocabs[2]

        train_iter = data.BucketIterator(train,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         sort=False,
                                         device=torch.device(device_set if torch.cuda.is_available() else "cpu"))

        dict_return = {'iters': [train_iter],
                       'vocabs': (inputs_word_document.vocab, inputs_char_document.vocab, labels.vocab)}

    return dict_return


# we need define full name vocab for some case with difference vector?
def make_vocab_all_task(path_folder_data_fake_full,
                        name_file_train_fake,
                        name_file_train_fake_2,
                        cache_folder,
                        name_vocab,
                        name_save_full_vocab,
                        min_freq_word=1,
                        min_freq_char=5):
    data = load_data_word_lstm_char(path_folder_data_fake_full,
                                    name_file_train_fake,
                                    name_file_train_fake_2,
                                    min_freq_word=min_freq_word,
                                    min_freq_char=min_freq_char,
                                    cache_folder=cache_folder,
                                    name_vocab=name_vocab)
    vocab_full = data['vocabs']
    with open(name_save_full_vocab, "wb") as wf:
        torch.save(vocab_full, wf)
