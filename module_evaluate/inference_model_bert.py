from transformers.configuration_bert import BertConfig

from module_train.bert_model.ZaloBert import BERTQa
from module_dataset.preprocess_dataset.handle_dataloader_bert import *
import torch


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


def get_predict_dl(model, tokenizer, args):
    list_id, list_para, list_test_sent = get_list_data_test_file(args.path_input_test_data, is_pair=True)

    list_predicts = []
    with open("test_submit2.txt", "w") as wf:
        for idx, e_sent in enumerate(list_test_sent):
            model.eval()
            e_sent = e_sent.replace("\n", "")
            id_test = list_id[idx]
            id_para = list_para[idx]

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

                predict_value = torch.max(logits, 1)[1]
                list_predict = predict_value.cpu().numpy().tolist()

                if list_predict[0] == 1:
                    wf.write("{},{}\n".format(id_test, id_para))

    return list_predicts


if __name__ == '__main__':
    class Args:
        do_lower_case = True
        folder_model = "../module_train/final_checkpoint/final_checkpoint_submit"

        path_input_test_data = "../module_dataset/dataset/dataset_preprocess/pair_sequence/test_data/" \
                               "private_test_pair_without_punc.csv"

        no_cuda = False
        n_gpu = 1
        device = "cuda:0"
        seed = 42

        max_seq_length = 400
        max_query_length = 64
        weight_class = [1, 1]


    args = Args()

    device = torch.device(args.device)
    tokenizer = BertTokenizer.from_pretrained(args.folder_model, do_lower_case=args.do_lower_case)

    config = BertConfig.from_pretrained(args.folder_model)

    # # custom some parameter for custom bert
    config = config.to_dict()
    config.update({"device": args.device})
    config = BertConfig.from_dict(config)

    model = BERTQa.from_pretrained(args.folder_model, config=config)

    model = model.to(device)
    get_predict_dl(model, tokenizer, args)
