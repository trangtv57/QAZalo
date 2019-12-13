import json
import time
from textblob import TextBlob


def translate_data(text, time_sleep=1.0):
    try:
        n_blob = TextBlob(str(text))
        str_vn_back_translate = n_blob.translate(from_lang='en', to='vi')
        time.sleep(time_sleep)
    except:
        return None

    return str(str_vn_back_translate)


def make_data_augment_squad_20(path_dict_question, path_dict_context):

    with open(path_dict_context, "r") as rf:
        dict_context = json.load(rf)

    with open("context_viet.csv", "a") as wf:
        for e_key, e_value in dict_context.items():
            e_value = e_value.replace("\t", "").replace("\n", "")
            # e_value = translate_data(e_value, time_sleep=1)
            print(e_value)
            if e_value is None:
                break
            else:
                line_write = '{}\t{}\n'.format(e_key, e_value)
                print(line_write)
                wf.write(line_write)
    time.sleep(10)
    print("done_context !!")

    with open(path_dict_question, "r") as rf_2:
        dict_question = json.load(rf_2)

    with open("question_viet_0_20.csv", "a") as wf_2:
        for e_key, e_value in dict_question.items():
            if int(e_key) > 0:
                for e_qs in e_value:
                    question, label = e_qs
                    question = question.replace("\t", "").replace("\n", "")
                    # question = translate_data(question, time_sleep=1)
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


if __name__ == '__main__':
    path_dict_question = "/..module_dataset/dataset/raw_dataset/qas_squad_dev_11.json"
    path_dict_context = "/..module_dataset/dataset/raw_dataset/context_squad_dev_11.json"
    make_data_augment_squad_20(path_dict_question, path_dict_context)
