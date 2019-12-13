import json
import glob
from collections import defaultdict


def get_dict_typing_error(path_file_dict):
    dict_typing_error = {}
    with open(path_file_dict, 'r') as rf:
        for e_line in rf.readlines():
            arr = e_line.replace("\n", "").split()
            dict_typing_error["{} ".format(arr[1])] = arr[0]

    return dict_typing_error


def get_list_from_file(path_word_list):
    list_word = []
    with open(path_word_list, "r") as rf:
        for e_line in rf.readlines():
            list_word.append(e_line.replace('\n', ''))
    return list_word


def load_config(path_file_config):
    with open(path_file_config, "r") as rf:
        cf = json.load(rf)
    return cf


def get_all_path_file_in_folder(path_folder):
    list_path_file = []
    path_folder_recursive = path_folder + "/**"
    for e_file in glob.glob(path_folder_recursive):
        list_path_file.append(e_file)
    return list_path_file


def get_name_folder_file(path_file):
    arr_path = path_file.split("/")
    path_folder = "/".join(arr_path[:-1])
    name_file = arr_path[-1]
    return path_folder, name_file


def get_data_from_index(list_data, list_index):
    n_list_data = []
    for e_index in list_index:
        n_list_data.append(list_data[e_index])
    return n_list_data


def get_dict_augment_data(path_file):
    dict_augment_data = defaultdict(list)

    with open(path_file, "r") as rf:
        for e_line in rf.readlines():
            arr_line = e_line.replace("\n", "").split("|")
            id_text = arr_line[0]
            text_data = arr_line[1]
            dict_augment_data[id_text].append(text_data)

    return dict_augment_data


def get_list_test_id_from_file(path_file):
    list_word = []
    list_id = []
    with open(path_file, "r") as rf:
        for e_line in rf.readlines():
            arr_line = e_line.replace("\n", "").split("|")
            list_id.append(arr_line[0])
            list_word.append(arr_line[1])
    return list_id, list_word
