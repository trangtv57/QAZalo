import os
from sklearn.model_selection import train_test_split
from module_dataset.preprocess_dataset.handle_text import *
from textblob import TextBlob
import time
from module_dataset.preprocess_dataset.handle_text import handle_text_qa, \
    handle_text_qa_with_segment, check_line_squad
from module_dataset.preprocess_dataset.utilities import *


def translate_data(text, time_sleep=1.0):
    try:
        n_blob = TextBlob(str(text))
        str_vn_back_translate = n_blob.translate(from_lang='en', to='vi')
        time.sleep(time_sleep)
    except:
        return None

    return str(str_vn_back_translate)


def load_data_full(path_train_json):
    l_tuple_question_document = []
    l_label = []
    with open(path_train_json, "r") as rf:
        data_full = json.load(rf)
        for e_data in data_full:
            # id = e_data['id']
            question = e_data['question'].replace("\t", " ")
            document = e_data['text'].replace("\t", " ")
            label = e_data['label']
            if label:
                label = "true"
            else:
                label = "false"
            l_tuple_question_document.append((question, document))
            l_label.append(label)
    return l_tuple_question_document, l_label


def make_dataset_one_and_pair(path_train_json, path_folder_one, path_folder_pair):
    l_tuple_question_document, l_label = load_data_full(path_train_json)

    data_full_train, data_test, l_label_full_train, l_label_test = \
        train_test_split(l_tuple_question_document, l_label, stratify=l_label, test_size=0.18)

    data_train, data_validation, l_label_train, l_label_validation = \
        train_test_split(data_full_train, l_label_full_train, stratify=l_label_full_train, test_size=0.15)

    train_one_not_sg = open(os.path.join(path_folder_one, "train_one_sequence_not_segment.csv"), "w")
    test_one_not_sg = open(os.path.join(path_folder_one, "test_one_sequence_not_segment.csv"), "w")
    validation_one_not_sg = open(os.path.join(path_folder_one, "validation_one_sequence_not_segment.csv"), "w")

    # train_one_has_sg = open(os.path.join(path_folder_one, "train_one_sequence_has_segment.csv"), "w")
    # test_one_has_sg = open(os.path.join(path_folder_one, "test_one_sequence_has_segment.csv"), "w")
    # validation_one_has_sg = open(os.path.join(path_folder_one, "validation_one_sequence_has_segment.csv"), "w")

    train_pair_not_sg = open(os.path.join(path_folder_pair, "train_pair_sequence_not_segment.csv"), "w")
    test_pair_not_sg = open(os.path.join(path_folder_pair, "test_pair_sequence_not_segment.csv"), "w")
    validation_pair_not_sg = open(os.path.join(path_folder_pair, "validation_pair_sequence_not_segment.csv"), "w")

    # train_pair_has_sg = open(os.path.join(path_folder_pair, "train_pair_sequence_has_segment.csv"), "w")
    # test_pair_has_sg = open(os.path.join(path_folder_pair, "test_pair_sequence_has_segment.csv"), "w")
    # validation_pair_has_sg = open(os.path.join(path_folder_pair, "validation_pair_sequence_has_segment.csv"), "w")

    for idx, e_tuple_train in enumerate(data_train):
        question, document = e_tuple_train

        document_not_segment = handle_text_qa(document).replace("\t", "")
        question_not_segment = handle_text_qa(question).replace("\t", "")

        # document_has_segment = handle_text_qa_with_segment(document).replace("\t", "")
        # question_has_segment = handle_text_qa_with_segment(question).replace("\t", "")

        line_write_one_not_sg = "{} ? {}\t{}\n".format(question_not_segment, document_not_segment, l_label_train[idx])
        train_one_not_sg.write(line_write_one_not_sg)

        # line_write_one_sg = "{} ? {}\t{}\n".format(question_has_segment, document_has_segment, l_label_train[idx])
        # train_one_has_sg.write(line_write_one_sg)

        line_write_pair_not_sg = "{}\t{}\t{}\n".format(question_not_segment, document_not_segment, l_label_train[idx])
        train_pair_not_sg.write(line_write_pair_not_sg)

        # line_write_pair_has_sg = "{}\t{}\t{}\n".format(question_has_segment, document_has_segment, l_label_train[idx])
        # train_pair_has_sg.write(line_write_pair_has_sg)

    for idx, e_tuple_test in enumerate(data_test):
        question, document = e_tuple_test

        document_not_segment = handle_text_qa(document).replace("\t", "")
        question_not_segment = handle_text_qa(question).replace("\t", "")

        # document_has_segment = handle_text_qa_with_segment(document).replace("\t", "")
        # question_has_segment = handle_text_qa_with_segment(question).replace("\t", "")

        line_write_one_not_sg = "{} ? {}\t{}\n".format(question_not_segment, document_not_segment, l_label_test[idx])
        test_one_not_sg.write(line_write_one_not_sg)

        # line_write_one_sg = "{} ? {}\t{}\n".format(question_has_segment, document_has_segment, l_label_test[idx])
        # test_one_has_sg.write(line_write_one_sg)

        line_write_pair_not_sg = "{}\t{}\t{}\n".format(question_not_segment, document_not_segment, l_label_test[idx])
        test_pair_not_sg.write(line_write_pair_not_sg)
        #
        # line_write_pair_has_sg = "{}\t{}\t{}\n".format(question_has_segment, document_has_segment, l_label_test[idx])
        # test_pair_has_sg.write(line_write_pair_has_sg)

    for idx, e_tuple_validation in enumerate(data_validation):
        question, document = e_tuple_validation

        document_not_segment = handle_text_qa(document).replace("\t", "")
        question_not_segment = handle_text_qa(question).replace("\t", "")

        # document_has_segment = handle_text_qa_with_segment(document).replace("\t", "")
        # question_has_segment = handle_text_qa_with_segment(question).replace("\t", "")

        line_write_one_not_sg = "{} ? {}\t{}\n".format(question_not_segment, document_not_segment, l_label_validation[idx])
        validation_one_not_sg.write(line_write_one_not_sg)

        # line_write_one_sg = "{} ? {}\t{}\n".format(question_has_segment, document_has_segment, l_label_validation[idx])
        # validation_one_has_sg.write(line_write_one_sg)

        line_write_pair_not_sg = "{}\t{}\t{}\n".format(question_not_segment, document_not_segment, l_label_validation[idx])
        validation_pair_not_sg.write(line_write_pair_not_sg)

        # line_write_pair_has_sg = "{}\t{}\t{}\n".format(question_has_segment, document_has_segment, l_label_validation[idx])
        # validation_pair_has_sg.write(line_write_pair_has_sg)


def save_dataset_squad_20(path_data_train_squad):
    dict_context = {}
    dict_question = defaultdict(list)

    w_train = open("dev_v3_viet_facebook_500_sent.csv", "a")
    with open(path_data_train_squad, "r", encoding="utf-8") as rf:
        data_squad_json = json.load(rf)
        count = 0
        for e_data in data_squad_json['data']:
            list_paragraphs = e_data['paragraphs']
            for e_paragraph in list_paragraphs:
                count += 1
                list_qas = e_paragraph['qas']
                context = e_paragraph['context']

                dict_context[count] = context
                for e_question in list_qas:
                    # print(e_question)

                    question = e_question['question']
                    print(question)
                    # if len(e_question['answers']) == 0:
                    #     print(e_question)
                    # # label = e_question['is_impossible']
                    # if label:
                    #     label = "false"
                    # elif label is not True:
                    #     label = "true"
                    # # with squad 1.1
                    label = "true"
                    question = question.replace("\n", "")
                    context = context.replace("\n", "")
                    line_write = "{}\t{}\t{}\n".format(question, context, label)
                    print(line_write)
                    w_train.write(line_write)
                    # dict_question[count].append((question, label))

    # with open("context_squad_train_11.json", "w") as wf:
    #     json.dump(dict_context, wf)
    #
    # with open("qas_squad_train_11.json", "w") as wf_qas:
    #     json.dump(dict_question, wf_qas)


def make_data_augment_squad_20(path_dict_question, path_dict_context):

    with open(path_dict_context, "r") as rf:
        dict_context = json.load(rf)
    with open(path_dict_question, "r") as rf_2:
        dict_question = json.load(rf_2)

    with open("context_viet.csv", "a") as wf:
        for e_key, e_value in dict_context.items():
            e_value = e_value.replace("\t", "").replace("\n", "")
            e_value = translate_data(e_value, time_sleep=0.5)
            print(e_value)
            if e_value is None:
                break
            else:
                line_write = '{}\t{}\n'.format(e_key, e_value)
                print(line_write)
                wf.write(line_write)
    time.sleep(10)
    print("done_context !!")

    with open("question_viet.csv", "a") as wf_2:
        for e_key, e_value in dict_question.items():
            for e_qs in e_value:
                question, label = e_qs
                question = question.replace("\t", "").replace("\n", "")
                question = translate_data(question, time_sleep=1)
                print(question)
                if label:
                    label = "true"
                else:
                    label = "false"
                if question is None:
                    break
                else:
                    line_write = '{}\t{}\t{}\n'.format(e_key, question, label)
                    wf_2.write(line_write)


def combine_handle_augment_squad_translate(path_document, path_question, path_folder_combine, squad_20=True):
    dict_document = {}
    dict_question = defaultdict(list)
    with open(path_document, "r") as rf:
        for e_line in rf.readlines():
            arr_e_line = e_line.replace("\n", "").split("\t")
            dict_document[arr_e_line[0]] = arr_e_line[1]

    with open(path_question, "r") as rf_question:
        for e_line in rf_question.readlines():
            arr_e_line = e_line.replace("\n", "").split("\t")
            dict_question[arr_e_line[0]].append((arr_e_line[1], arr_e_line[2]))

    # data_one_has_segment = open(os.path.join(path_folder_combine, "squad_dev_one_has_sg_20.csv"), "w")
    data_one_not_segment = open(os.path.join(path_folder_combine, "squad_dev_one_not_sg_11.csv"), "w")

    # data_pair_has_segment = open(os.path.join(path_folder_combine, "squad_dev_pair_has_sg_20.csv"), "w")
    data_pair_not_segment = open(os.path.join(path_folder_combine, "squad_dev_pair_not_sg_11.csv"), "w")

    for e_key, e_content_document in dict_document.items():
        document_not_segment = handle_text_qa(e_content_document)
        # document_has_segment = handle_text_qa_with_segment(e_content_document)

        if e_key in dict_question:
            list_question_answer = dict_question[e_key]
            for e_qas in list_question_answer:
                question, label = e_qas
                # print(type(label))
                if squad_20:
                    if label == "true":
                        label = "false"
                    else:
                        label = "true"
                question_not_sg = handle_text_qa(question)
                question_has_sg = handle_text_qa_with_segment(question)
                print(question_has_sg)

                # line_write_one_has_sg = "{} {}\t{}\n".format(question_has_sg, document_has_segment, label)
                # data_one_has_segment.write(line_write_one_has_sg)

                line_write_one_not_sg = "{} {}\t{}\n".format(question_not_sg, document_not_segment, label)
                data_one_not_segment.write(line_write_one_not_sg)

                # line_write_pair_has_sg = "{}\t{}\t{}\n".format(question_has_sg, document_has_segment, label)
                # data_pair_has_segment.write(line_write_pair_has_sg)

                line_write_pair_not_sg = "{}\t{}\t{}\n".format(question_not_sg, document_not_segment, label)
                data_pair_not_segment.write(line_write_pair_not_sg)


def convert_test_data_to_right_format(path_file_test_json, name_process_file_test,
                                      is_segment=True, is_pair_sequence=True):
    with open(name_process_file_test, "w") as wf:
        with open(path_file_test_json, "r") as rf:
            data_full = json.load(rf)
            i = 0
            for e_data in data_full:
                id_test = e_data['__id__']
                question = e_data['question'].replace("\t", " ")

                if is_segment:
                    question = handle_text_qa_with_segment(question)
                else:
                    question = handle_text_qa(question)

                list_paragprahs = e_data['paragraphs']
                for e_para in list_paragprahs:
                    id_para = e_para['id']
                    context_para = e_para['text']

                    if is_segment:
                        context_para = handle_text_qa_with_segment(context_para)
                    else:
                        context_para = handle_text_qa(context_para)

                    if is_pair_sequence:
                        line_write_seq = "{}\t{}\t{}\t{}\n".format(id_test, id_para, question, context_para)
                        wf.write(line_write_seq)
                    else:
                        line_write_one = "{}\t{}\t{} {}\n".format(id_test, id_para, question, context_para)
                        wf.write(line_write_one)
                    i += 1


def convert_pair_task_to_one(path_file):
    path_file_one_sequence = path_file.replace(".csv", "_one_sequence.csv")
    with open(path_file_one_sequence, 'w') as wf:
        with open(path_file, "r") as rf:
            for e_line in rf.readlines():
                e_line = e_line.replace("\n", "")
                arr_e_line = e_line.split("\t")
                line_write = "{} {}\t{}\n".format(arr_e_line[0], arr_e_line[1], arr_e_line[2])
                wf.write(line_write)


def filter_squad_data(path_folder_squad):
    list_file_squad = get_all_path_file_in_folder(path_folder_squad)
    for e_file in list_file_squad:
        path_squad_new = e_file.replace(".csv", "_with_filter.csv")
        with open(path_squad_new, "w") as wf:
            with open(e_file, "r") as rf:
                for e_line in rf.readlines():
                    arr_e_line = e_line.replace("\n", "").split("\t")
                    result_check = check_line_squad(arr_e_line[0], arr_e_line[1])
                    if result_check:
                       wf.write(e_line)


def check_train_2(path_train_2):
    new_path_2 = path_train_2.replace(".csv", "_with_check.csv")
    with open(new_path_2, "w") as wf:
        with open(path_train_2, "r") as rf:
            for e_line in rf.readlines():
                if len(e_line.split("\t")) == 3:
                    wf.write(e_line)


def make_data_for_build_lm(path_file):
    n_path_file = path_file.replace(".txt", "_norm.csv")
    print(n_path_file)
    with open(n_path_file, "w") as wf:
        with open(path_file, "r") as rf:
            for e_line in rf.readlines():
                e_line = e_line.replace("\n", "")
                e_line = handle_text_qa_with_segment(e_line)
                wf.write(e_line + "\n")


def make_data_squad_english(path_file_context, path_file_question):
    with open(path_file_context, "r") as rf_context:
        dict_context = json.load(rf_context)
    with open(path_file_question, "r") as rf_question:
        dict_question = json.load(rf_question)

    with open("squad_dev_english_20_pair.csv", "w") as wf:
        for e_key, e_value in dict_context.items():
            list_question = dict_question[e_key]
            for e_question_label in list_question:
                question = e_question_label[0]
                label = e_question_label[1]
                if label:
                    label = "true"
                else:
                    label = "false"

                sample_line = "{}\t{}\t{}\n".format(question, e_value, label)
                wf.write(sample_line)


def get_data_squad_with_condition(path_squad):
    w_squad = open("squad_train_20_sent_lt_4.csv", "w")
    with open(path_squad, "r") as rf:
        for e_line in rf.readlines():
            e_line = e_line.replace("\n", "")
            arr_e_line = e_line.split("\t")

            e_document = arr_e_line[1]

            sent = sent_tokenize(e_document)
            if len(sent) < 4:
                w_squad.write(e_line + "\n")


def update_kw_is_impossible(path_json_1):
    with open(path_json_1, "r", encoding="utf-8") as rf:
        json_1 = json.load(rf)

    n_line = len(json_1['data'])
    for i in range(n_line):
        n_paragraphs = len(json_1['data'][i]['paragraphs'])
        for j in range(n_paragraphs):
            len_number_question_per_graphs = len(json_1['data'][i]['paragraphs'][j]['qas'])
            for k in range(len_number_question_per_graphs):
                json_1['data'][i]['paragraphs'][j]['qas'][k].update({'is_impossible':False})

    with open("fb_dev_v2.json", "w") as wf:
        json.dump(json_1, wf)




def combine_json_data(json_data_1, json_data_2):
    with open(json_data_1, "r", encoding="utf-8") as rf:
        json_1 = json.load(rf)

    with open(json_data_2, "r", encoding="utf-8") as rf_2:
        json_2 = json.load(rf_2)

    #
    print(type(json_1['data']))
    print(type(json_2['data']))

    print(json_1['data'][0])
    print(json_2['data'][0])
    #
    data_1 = json_1['data']
    data_1.extend(json_2['data'])
    dict_full = {"data": data_1}
    with open("combine_train_viet_test_dev_fb.json", "w") as wf:
        json.dump(dict_full, wf)

    for e_data in data_1:
        list_paragraphs = e_data['paragraphs']
        # print(list_paragraphs[0:2])
        for e_paragraph in list_paragraphs:
            list_qas = e_paragraph['qas']
            context = e_paragraph['context']
            print(list_qas)
            print(context)
            break


def split_file_validation(path_validation):
    list_content = []
    l_label = []
    with open(path_validation, "r") as rf:
        for e_line in rf.readlines():
            e_line = e_line.replace("\n", "")
            arr_e_line = e_line.split("\t")
            e_question_context = arr_e_line[0] + "\t" + arr_e_line[1]
            e_label = arr_e_line[2]

            list_content.append(e_question_context)
            l_label.append(e_label)

    content_train, content_test, label_train, label_test = \
        train_test_split(list_content, l_label, stratify=l_label, test_size=0.5)

    with open("val_origin_train_ensemble.csv", "w") as wf:
        for idx, e_content_train in enumerate(content_train):
            line_write = "{}\t{}\n".format(e_content_train, label_train[idx])
            wf.write(line_write)

    with open("val_origin_test_ensemble.csv", "w") as wf:
        for idx, e_content_test in enumerate(content_test):
            line_write = "{}\t{}\n".format(e_content_test, label_test[idx])
            wf.write(line_write)


if __name__ == '__main__':
    path_train_json = "../module_dataset/dataset/raw_dataset/test-context-vi-question-vi.json"
    path_folder_one = "../module_dataset/dataset/dataset_split_with_preprocess/" \
                      "one_sequence_2"
    path_folder_pair = "../module_dataset/dataset/dataset_split_with_preprocess/" \
                       "pair_sequence_2"
    # make_dataset_one_and_pair(path_train_json, path_folder_one, path_folder_pair)
    path_data_squad = "/home/trangtv/Downloads/max_pooling/MLQA_V1/dev/dev-context-vi-question-vi.json"
    # save_dataset_squad_20(path_data_squad)
    path_dict_context = "../moudle_dataset/dataset/raw_dataset/context_squad.json"
    path_dict_question = "../moudle_dataset/dataset/raw_dataset/qas_squad.json"
    # make_data_augment_squad_20(path_dict_question, path_dict_context)

    path_context_squad_translate = "../module_dataset/dataset/new_augment_dataset/" \
                                   "context_viet_dev_11.csv"
    path_question_squad_translate = "../module_dataset/dataset/new_augment_dataset/" \
                                    "question_viet_dev_11.csv"
    path_combine_squad = "../module_dataset/dataset/augment_dataset_large/"
    # combine_handle_augment_squad_translate(path_context_squad_translate, path_question_squad_translate,
    #                                        path_combine_squad, squad_20=False)

    path_file_test_json = "../module_dataset/dataset/raw_dataset/test_private.json"
    convert_test_data_to_right_format(path_file_test_json, "private_test_pair_without_punc.csv",
                                      is_segment=False, is_pair_sequence=True)
    path_squad = "../module_dataset/dataset/augment_dataset_large/"
    # filter_squad_data(path_squad)
    path_context = "../module_dataset/dataset/raw_dataset/context_squad_dev_20.json"
    path_question = "../module_dataset/dataset/raw_dataset/qas_squad_dev_20.json"
    # make_data_squad_english(path_context, path_question)
    # get_data_squad_with_condition("../module_dataset/dataset/augment_dataset_large/squad_train_pair_not_sg_20_with_filter.csv")
    # update_kw_is_impossible(path_json_2)
    path_json_1 = "../module_dataset/preprocess_dataset/combine_train_viet_test_fb.json"
    path_json_2 = "../module_dataset/preprocess_dataset/fb_dev_v2.json"
    combine_json_data(path_json_1, path_json_2)
    # split_file_validation("../module_dataset/dataset/dataset_split_with_preprocess/pair_sequence/validation_pair_sequence_not_segment.csv")
