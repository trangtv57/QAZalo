from module_train.basic_model.model_architecture_dl.model_one_sequence_learning import LSTMCNNWord
from module_train.basic_model.model_architecture_dl.model_pair_bdaf import BiDAF
from module_train.basic_model.model_architecture_dl.model_pair_infersent import LSTMCNNWordInferSent

from torchtext import data
import torch


def get_input_processor_words(inputs, type_model, vocab_word, vocab_char):
    if "one_sequence" in type_model:

        inputs_word = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True, include_lengths=True)

        inputs_char_nesting = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>", batch_first=True)

        inputs_char = data.NestedField(inputs_char_nesting,
                                       init_token="<bos>", eos_token="<eos>")

        inputs_word.vocab = vocab_word
        inputs_char.vocab = inputs_char_nesting.vocab = vocab_char
        fields = [(('inputs_word', 'inputs_char'), (inputs_word, inputs_char))]

        if not isinstance(inputs, list):
            inputs = [inputs]

        examples = []

        for line in inputs:
            examples.append(data.Example.fromlist([line], fields))

        dataset = data.Dataset(examples, fields)
        batchs = data.Batch(data=dataset,
                            dataset=dataset,
                            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    else:
        inputs_word_query = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True, include_lengths=True)
        inputs_char_query_nesting = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>", batch_first=True)
        inputs_char_query = data.NestedField(inputs_char_query_nesting, init_token="<bos>", eos_token="<eos>")

        inputs_word_document = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True, include_lengths=True)
        inputs_char_document_nesting = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>",
                                                  batch_first=True)
        inputs_char_document = data.NestedField(inputs_char_document_nesting, init_token="<bos>", eos_token="<eos>")

        fields = ([(('inputs_word_query', 'inputs_char_query'), (inputs_word_query, inputs_char_query)),
                   (('inputs_word_document', 'inputs_char_document'), (inputs_word_document, inputs_char_document))])

        inputs_word_query.vocab = inputs_word_document.vocab = vocab_word
        inputs_char_query.vocab = inputs_char_query_nesting.vocab = \
            inputs_char_document_nesting.vocab = inputs_char_document.vocab = vocab_char

        # print(vocab_word.stoi)
        # print(vocab_char.stoi)

        if not isinstance(inputs, list):
            inputs = [inputs]

        examples = []

        for line in inputs:
            tuple_line = line.split("\t")
            example = data.Example.fromlist(tuple_line, fields)
            examples.append(example)

        dataset = data.Dataset(examples, fields)
        batchs = data.Batch(data=dataset,
                            dataset=dataset,
                            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    # Entire input in one batch
    return batchs


def get_list_data_test_file(path_data_test, is_pair=True):
    l_id = []
    l_para = []
    l_content = []
    with open(path_data_test, "r") as rf:
        for e_line in rf.readlines():
            e_line = e_line.replace("\n", "")
            arr_e_line = e_line.split("\t")
            if is_pair:
                id_test = arr_e_line[0]
                id_para = arr_e_line[1]
                question_data = arr_e_line[2]
                document_data = arr_e_line[3]
                content_data = question_data + "\t" + document_data
            else:
                id_test = arr_e_line[0]
                id_para = arr_e_line[1]
                content_data = arr_e_line[2]
            l_id.append(id_test)
            l_para.append(id_para)
            l_content.append(content_data)

    return l_id, l_para, l_content


def get_predict_dl(path_data_test, path_save_model, path_model_checkpoint,
                   is_pair=False,
                   type_model='one_sequence_lstm_cnn'):
    list_id, list_para, list_test_sent = get_list_data_test_file(path_data_test, is_pair=is_pair)

    if type_model == "one_sequence_lstm_cnn":
        model = LSTMCNNWord.load(path_save_model, path_model_checkpoint)

    elif type_model == "pair_sequence_infer_sent":
        model = LSTMCNNWordInferSent.load(path_save_model, path_model_checkpoint)

    elif type_model == "pair_sequence_bidaf":
        model = BiDAF.load(path_save_model, path_model_checkpoint)

    vocab_word, vocab_char, vocab_label = model.vocabs
    print(vocab_char.stoi)
    # print(vocab_word.stoi)
    # print(vocab_label.stoi)

    list_predicts = []
    with open("test_submit2.txt", "w") as wf:
        for idx, e_sent in enumerate(list_test_sent):

            data_e_line = get_input_processor_words(e_sent, type_model, vocab_word, vocab_char)
            predict = model(data_e_line)
            # print(predict.cpu().numpy().tolist()[0])
            predict_value = torch.max(predict, 1)[1].cpu().numpy().tolist()[0]
            # print(predict_value)
            if predict_value != 0:
                predict_list = predict.cpu().numpy().tolist()[0]
                line_write = "{},{}\n".format(list_id[idx], list_para[idx])
                wf.write(line_write)
            list_predicts.append(predict)

    return list_predicts


if __name__ == '__main__':
    path_test = "../module_dataset/dataset/process_train_test/test_pair_has_segment.csv"

    path_save_model = "../module_train/save_model/model_2/"
    path_model_checkpoint = "../module_train/save_model/model_2/" \
                            "pair_bidaf_has_sg__epoch_1_train_loss_0.633_acc0.657_f1_0.731_test_loss_0.73_acc0.571_f1_0.672"
    get_predict_dl(path_test, path_save_model, path_model_checkpoint,
                   is_pair=True,
                   type_model="pair_sequence_bidaf"
                   )
