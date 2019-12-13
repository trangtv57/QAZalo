from module_train.basic_model.model_architecture_dl.model_one_sequence_learning import LSTMCNNWord
from module_train.basic_model.model_architecture_dl.model_pair_infersent import LSTMCNNWordInferSent
from module_train.basic_model.model_architecture_dl.model_pair_bdaf import BiDAF

from module_train.basic_model.train_basic import Trainer
from module_dataset.preprocess_dataset.handle_dataloader_basic import *

from transformers.tokenization_bert import BertTokenizer
from transformers.configuration_bert import BertConfig
from module_train.bert_model.train_bert import *
from module_train.bert_model.ZaloBert import BERTQa


def train_model_basic(cf_common, cf_model):
    path_data = cf_common['path_data']
    path_data_train = cf_common['path_data_train']
    path_data_test = cf_common['path_data_test']

    type_model = cf_common['type_model']
    model = None
    data_train_iter = None
    data_test_iter = None
    data = None

    # first load data
    if "one_sequence" in type_model:
        data = load_data_word_lstm_char(path_data,
                                        path_data_train,
                                        path_data_test,
                                        device_set=cf_common['device_set'],
                                        min_freq_word=cf_common['min_freq_word'],
                                        min_freq_char=cf_common['min_freq_char'],
                                        batch_size=cf_common['batch_size'],
                                        cache_folder=cf_common['cache_folder'],
                                        name_vocab=cf_common['name_vocab'],
                                        path_vocab_pre_built=cf_common['path_vocab_pre_built']
                                        )

        data_train_iter = data['iters'][0]

        if path_data_test is not None:
            data_test_iter = data['iters'][1]
        else:
            data_test_iter = None

    if "pair_sequence" in type_model:
        data = load_data_pair_task(path_data,
                                   path_data_train,
                                   path_data_test,
                                   device_set=cf_common['device_set'],
                                   min_freq_word=cf_common['min_freq_word'],
                                   min_freq_char=cf_common['min_freq_char'],
                                   batch_size=cf_common['batch_size'],
                                   cache_folder=cf_common['cache_folder'],
                                   name_vocab=cf_common['name_vocab'],
                                   path_vocab_pre_built=cf_common['path_vocab_pre_built']
                                   )

        data_train_iter = data['iters'][0]

        if path_data_test is not None:
            data_test_iter = data['iters'][1]
        else:
            data_test_iter = None

    print("!!Load dataset done !!\n")

    if type_model == "one_sequence_lstm_cnn":
        model = LSTMCNNWord.create(cf_common['path_save_model'] + cf_common['folder_model'],
                                   cf_model,
                                   data['vocabs'],
                                   device_set=cf_common['device_set'])

    elif type_model == "pair_sequence_infer_sent":
        model = LSTMCNNWordInferSent.create(cf_common['path_save_model'] + cf_common['folder_model'],
                                            cf_model,
                                            data['vocabs'],
                                            device_set=cf_common['device_set'])

    elif type_model == "pair_sequence_bidaf":
        model = BiDAF.create(cf_common['path_save_model'] + cf_common['folder_model'],
                                            cf_model,
                                            data['vocabs'],
                                            device_set=cf_common['device_set'])

    trainer = Trainer(cf_common['path_save_model'] + cf_common['folder_model'],
                      model,
                      cf_model,
                      cf_common['prefix_model'],
                      cf_common['log_file'],
                      len(data['vocabs'][2]),
                      data_train_iter,
                      data_test_iter)

    trainer.train(cf_common['num_epochs'])


def train_model_bert(args):
    # need remake config with device option for train with another cuda device
    config = BertConfig.from_pretrained(args.folder_model)

    config = config.to_dict()
    config.update({"device": args.device})
    config.update({"use_pooler": args.use_pooler})
    config.update({"weight_class": args.weight_class})
    config.update({"output_hidden_states": args.output_hidden_states})
    config = BertConfig.from_dict(config)

    tokenizer = BertTokenizer.from_pretrained(args.folder_model)
    model = BERTQa.from_pretrained(args.folder_model, config=config)
    model = model.to(args.device)
    train_squad(args, tokenizer, model)


if __name__ == '__main__':
    cf_common = {
        "path_save_model": "save_model/",
        "path_data": "../module_dataset/dataset/dataset_split_with_preprocess/pair_sequence",
        "path_data_train": "train_origin_has_segment.csv",
        "path_data_test": "test_pair_sequence_has_segment.csv",
        "prefix_model": "pair_sequence_bidaf_has_sg_",
        "log_file": "log_pair_sequence_bidaf_has_sg_hsw_64_char_emb_64_drop_03.txt",
        "type_model": "pair_sequence_bidaf",
        "folder_model": "model_1",
        'path_checkpoint': "",
        "device_set": "cuda:0",
        "num_epochs": 30,
        "min_freq_word": 1,
        "min_freq_char": 5,
        "path_vocab_pre_built": "../module_dataset/dataset/vocab_build_all/vocab_baomoi_400_augment_has_sg_min_freq_2.pt",
        "cache_folder": None,
        "name_vocab": None,
        "sort_key": True,
        "batch_size": 32
    }

    cf_model_lstm_cnn_word = {
        'use_xavier_weight_init': True,
        'word_embedding_dim': 400,
        'char_embedding_dim': 32,
        'hidden_size_word': 32,
        'hidden_size_char_lstm': 16,
        'use_highway_char': False,
        'use_char_cnn': False,
        'dropout_cnn_char': 0.3,
        'D_cnn': '1_D',
        'char_cnn_filter_num': 5,
        'char_window_size': [2, 3],
        'use_last_as_ft': False,
        "option_last_layer": "max_pooling",
        "cnn_filter_num": 16,
        "window_size": [1],
        'dropout_cnn_word': 0.55,
        'dropout_rate': 0.35,
        'learning_rate': 0.0001,
        'weight_decay': 0
    }

    cf_model_infer_sent = {
        'use_xavier_weight_init': True,
        'word_embedding_dim': 400,
        'char_embedding_dim': 32,
        'hidden_size_word': 32,
        'hidden_size_char_lstm': 16,

        'use_highway': False,
        'use_char_cnn': False,
        'dropout_cnn_char': 0.3,
        'D_cnn': '1_D',
        'char_cnn_filter_num': 5,
        'char_window_size': [2, 3],

        "option_last_layer": "max_pooling",
        "cnn_filter_num": 16,
        "window_size": [1],
        'dropout_cnn_word': 0.55,

        'hidden_layer_1_dim': 1600,
        'hidden_layer_2_dim': 512,

        'weight_class': [1, 1],
        'dropout_rate': 0.55,
        'learning_rate': 0.0001,
        'weight_decay': 0
    }

    cf_model_bidaf = {
        'use_xavier_weight_init': True,
        'word_embedding_dim': 400,
        'char_embedding_dim': 64,
        'hidden_size_word': 64,
        'hidden_size_char_lstm': 8,

        'use_highway': True,
        'use_char_cnn': True,
        'dropout_cnn_char': 0.3,
        'char_cnn_filter_num': 1,
        'char_window_size': [3],

        "use_modeling_in_last": False,
        'num_layer_modeling_lstm': 2,
        "option_last_layer": "max_pooling",
        "cnn_filter_num": 2,
        "window_size": [3],
        'dropout_cnn': 0.55,

        'weight_class': [0.6, 1],
        'dropout_rate': 0.3,
        'learning_rate': 0.0001,
        'weight_decay': 0
    }

    # train_model_basic(cf_common, cf_model_bidaf)

    class Args:
        do_lower_case = True
        folder_model = 'checkpoint_tune_squad_viet/checkpoint_after_tune_squad_viet'

        path_input_train_data = "../module_dataset/dataset/dataset_preprocess/pair_sequence/train_data/" \
                                "train_test_origin_1k_dev.csv"
        path_input_test_data = "../module_dataset/dataset/dataset_preprocess/pair_sequence/train_data/val_origin_1k.csv"
        path_input_validation_data = None

        load_data_from_pt = False
        path_pt_train_dataset = "../module_dataset/dataset/dataset_preprocess/train_test_origin_1k_dev.pt"
        path_pt_test_dataset = "../module_dataset/dataset/dataset_preprocess/val_origin_1k.pt"
        path_pt_validation_dataset = None

        path_log_file = "save_model/log_file_train_origin_1k_dev.txt"
        output_dir = "save_model/"

        max_seq_length = 400
        max_query_length = 64

        batch_size = 6

        num_labels = 2
        weight_class = [1, 1]

        learning_rate = 1e-5
        gradient_accumulation_steps = 5
        weight_decay = 0.0
        adam_epsilon = 1e-8
        max_grad_norm = 1.0
        warmup_steps = 0
        use_pooler = True
        output_hidden_states = True
        # if use_pooler= False (mean concat 4 CLS in 4 last hidden_state BERT)
        # you need to set output_hidden_states=True.

        num_train_epochs = 5
        save_steps = int(300 / gradient_accumulation_steps)

        no_cuda = False
        n_gpu = 1
        device = "cuda:0"
        seed = 42

    args = Args()

    train_model_bert(args)
