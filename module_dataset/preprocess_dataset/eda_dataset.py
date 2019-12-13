from module_dataset.preprocess_dataset.handle_text import *
from transformers.tokenization_bert import BertTokenizer
from collections import defaultdict
import torch
import torch.nn.functional as F


def show_length_of_document(path_document_origin, path_document_augment, path_file_test):
    list_len_origin = []
    list_len_augment = []
    l_label_origin = []
    l_label_augment = []

    with open(path_document_origin, "r") as r_origin:
        for e_line in r_origin.readlines():
            e_line = e_line.replace("\n", "")
            arr_e_line = e_line.split("\t")
            if len(arr_e_line) == 3:

                # print(arr_e_line)
                len_document = len(arr_e_line[1].split(" "))
                # if len_document == 1:
                #     print(arr_e_line)
                l_label_origin.append(arr_e_line[2])
                list_len_origin.append(len_document)

    print(l_label_origin[0:10])
    print(l_label_origin.count("true"))
    print(l_label_origin.count("false"))

    with open(path_document_augment, "r") as r_augment:
        for e_line in r_augment.readlines():
            e_line = e_line.replace("\n", "")
            arr_e_line = e_line.split("\t")
            # print(arr_e_line)
            len_document = len(arr_e_line[1].split(" "))
            l_label_augment.append(arr_e_line[2])
            list_len_augment.append(len_document)
    print(l_label_augment[0:10])
    print(l_label_augment.count("true"))
    print(l_label_augment.count("false"))

    list_len_test = []
    with open(path_file_test, "r") as r_test:
        for e_line in r_test.readlines():
            arr_e_line = e_line.split("\t")
            len_document = len(arr_e_line[3].split(" "))
            list_len_test.append(len_document)

    # sorted(list_len_origin)
    # sorted(list_len_augment)
    print(sorted(list_len_origin))
    print(sorted(list_len_augment))
    print(sorted(list_len_test))
    print(len(sorted(list_len_origin)))
    print(len(sorted(list_len_augment)))
    print(len(sorted(list_len_test)))
    print("min_len document origin: ", min(list_len_origin))
    print("min_len document augment: ", min(list_len_augment))

    print("max_len document origin: ", max(list_len_origin))
    print("max_len document augment: ", max(list_len_augment))

    print("mean_len document origin: ", sum(list_len_origin)/len(list_len_origin))
    print("mean_len document augment: ", sum(list_len_augment)/len(list_len_augment))


def check_length_test(path_test_pair, index_query=2, index_document=3, tokenizer=None):
    with open(path_test_pair, "r") as rf:
        count = 0
        list_document = []
        list_full_query_document = []
        for e_line in rf.readlines():
            e_line = e_line.replace("\n", "")
            arr_e_line = e_line.split("\t")

            e_query = e_line[index_query]
            e_document = arr_e_line[index_document]
            list_document.append(e_document)

            sent = sent_tokenize(e_document)

            query_document = "{} {}".format(e_query, e_document)

            if len(sent) < 3:
                print(query_document)

            list_full_query_document.append(query_document)
            count += 1
    #
    # print("tong so document: ", count)
    # list_unique_document = list(set(list_document))
    # print("tong so unique document la: ", len(list_unique_document))
    #
    # # check length with token split
    # dict_len_sent = defaultdict(lambda: 0)
    # for e_document in list_full_query_document:
    #     # print(e_document)
    #     # split sent
    #     sent = sent_tokenize(e_document)
    #     len_sent = len(sent)
    #     if len_sent < 3:
    #         print(e_document)
    #     dict_len_sent[len_sent] += 1
    #
    # print("len statistical of context document: \n", dict_len_sent)
    #
    # if tokenizer is not None:
    #     dict_lent_token_bert = defaultdict(lambda :0)
    #     for i, e_document_query in enumerate(list_full_query_document):
    #         token_bert = tokenizer.tokenize(e_document_query)
    #         if i == 0:
    #             print(e_document_query)
    #             print(token_bert)
    #         len_token_bert = len(token_bert)
    #         dict_lent_token_bert[len_token_bert] += 1
    #
    #     print("len token bert of both query and document: \n", dict_lent_token_bert)


def get_list_data_test_file(path_data_test, is_pair=True):
    l_label = []
    l_content = []
    with open(path_data_test, "r") as rf:
        for e_line in rf.readlines():
            e_line = e_line.replace("\n", "")
            arr_e_line = e_line.split("\t")
            if is_pair:
                  #                 id_test = arr_e_line[0]
                #                 id_para = arr_e_line[1]
                question_data = arr_e_line[0]
                document_data = arr_e_line[1]
                label = arr_e_line[2]
                content_data = question_data + "\t" + document_data

            l_label.append(label)
            l_content.append(content_data)

    return l_label, l_content


def eda_get_wrong_predict(path_data, model, tokenizer, args):
    l_label, list_test_sent = get_list_data_test_file(args.path_input_test_data, is_pair=True)

    list_predicts = []
    with open("wrong_validation_origin.txt", "w") as wf:
        for idx, e_sent in enumerate(list_test_sent):
            model.eval()
            e_sent = e_sent.replace("\n", "")
            question_text = e_sent.split("\t")[0]
            document_text = e_sent.split("\t")[1].split(" ")

            example = SquadExample(question_text=question_text,
                                   doc_tokens=document_text,
                                   is_has_answer=True)
            examples = [example]
            features = convert_examples_to_features(examples, tokenizer,
                                                    args.max_seq_length,
                                                    args.max_query_length)
            #             print(features[0].input_ids)
            #             print(type(features[0].input_ids))
            ft_input_ids = torch.tensor([e_feature.input_ids for e_feature in features], dtype=torch.long).to(
                args.device)
            ft_input_mask = torch.tensor([e_feature.input_mask for e_feature in features], dtype=torch.long).to(
                args.device)
            ft_segment_ids = torch.tensor([e_feature.segment_ids for e_feature in features], dtype=torch.long).to(
                args.device)
            with torch.no_grad():
                logits = model.forward(ft_input_ids,
                                       ft_input_mask,
                                       ft_segment_ids)

                output_softmax = F.softmax(logits)

                if torch.argmax(output_softmax) == 1:
                    value_predict = torch.max(output_softmax)
                else:
                    value_predict = torch.min(output_softmax)

                predict_round = torch.argmax(logits).cpu().numpy()

                if l_label[idx] == "true":
                    label_number = 1
                else:
                    label_number = 0

                if predict_round != label_number:
                    line_write = "{}\t{}\t{}\n".format(l_label[idx], value_predict, list_test_sent[idx])
                    print(line_write)
                    wf.write(line_write)
    return list_predicts


if __name__ == '__main__':

    path_origin = "/..module_dataset/dataset/dataset_split_with_preprocess/pair_sequence/tain_test_origin_viet_fb.csv"
    path_augment = "/..module_dataset/dataset/dataset_split_with_preprocess/pair_sequence/train_pair_sequence_not_segment.csv"
    path_test = "/..module_dataset/dataset/process_train_test/test_pair_has_segment.csv"
    path_bert_uncase = "/..module_dataset/preprocess_dataset/cache_bert_uncase"
    show_length_of_document(path_origin, path_augment, path_test)
    # a = [525, 525, 525, 525, 527, 527, 527, 527, 527, 527, 527, 527, 527, 527, 527, 527, 527, 527, 527, 527, 527, 527, 527, 527, 530, 530, 530, 530, 530, 530, 530, 530, 530, 530, 533, 533, 533, 533, 533, 536, 536, 536, 536, 536, 541, 541, 541, 541, 541, 541, 541, 541, 541, 541, 544, 544, 544, 544, 544, 546, 546, 546, 546, 546, 547, 547, 547, 551, 561, 561, 561, 561, 561, 568, 568, 568, 568, 568, 573, 573, 573, 573, 573, 573, 573, 573, 573, 575, 575, 575, 575, 575, 575, 575, 575, 575, 575, 575, 575, 575, 578, 578, 578, 578, 578, 587, 587, 587, 587, 587, 587, 587, 587, 587, 597, 597, 597, 597, 597, 597, 597, 597, 597, 597, 629, 629, 629, 629, 629, 630, 630, 630, 630, 630, 630, 630, 630, 630, 630, 657, 657, 657, 657, 657, 657, 657, 657, 657, 657, 699, 699, 699, 699, 699, 699, 699, 699, 701, 701, 701, 701, 701, 701, 701, 701, 701, 701, 714, 714, 714, 714, 714, 833, 833, 833, 833, 833, 833, 833, 833, 833, 833, 833, 833, 833, 833, 833]
    # print(len(a))
    path_test_submit = "/..module_dataset/dataset/augment_dataset_large/squad_train_pair_not_sg_20_with_filter.csv"

    # tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased", cache_dir="cache_bert_uncase/",
    #                                           do_lower_case=True)

    # check_length_test(path_test_submit, index_query=0, index_document=1, tokenizer=None)

