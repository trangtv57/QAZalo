from transformers.tokenization_bert import BertTokenizer
from random import shuffle
import torch

from torch.utils.data import TensorDataset, DataLoader, RandomSampler

# file data loader custom from util_squad in hugging face transfomrers
class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 question_text,
                 doc_tokens,
                 is_has_answer=None):
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.is_has_answer = is_has_answer

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        s += ", is_has_answer: %r" % (self.is_has_answer)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 example_index,
                 tokens,
                 token_to_orig_map,
                 input_ids,
                 input_mask,
                 segment_ids,
                 is_has_answer=None):
        self.example_index = example_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_has_answer = is_has_answer


def read_squad_example_from_file(input_data, is_training=True):
    with open(input_data, "r") as rf:
        examples = []
        for e_line in rf.readlines():
            e_line = e_line.replace("\n", "")
            arr_e_line = e_line.split("\t")

            if is_training:
                if arr_e_line[2] == "true":
                    is_has_answer = 1
                else:
                    is_has_answer = 0
            else:
                is_has_answer = None

            question_text = arr_e_line[0]
            doc_token = arr_e_line[1].split(" ")

            example = SquadExample(question_text=question_text,
                                   doc_tokens=doc_token,
                                   is_has_answer=is_has_answer)
            examples.append(example)

    return examples


# TODO we can add doc stride again for using ensemble for multi paragpraph
def convert_examples_to_features(examples, tokenizer, max_seq_length, max_query_length):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        # tok to orig index: using for remember the token has split by piece
        # like "trang la ta" => "tra #ng la ta" => [0, 0, 1, 2] 0, 0 mean 2 first token belong just first token
        # orig to tok index using for save number length of each token
        # all doc tokens for save all sub token of context
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        # The -3 accounts for [CLS], [SEP] and [SEP] max seq length is length of context.
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        len_context = len(all_doc_tokens)
        if len_context > max_tokens_for_doc:
            len_context = max_tokens_for_doc

        tokens = []
        token_to_orig_map = {}

        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for i in range(0, len_context):
            token_to_orig_map[len(query_tokens) + 2 + i] = tok_to_orig_index[i]
            tokens.append(all_doc_tokens[i])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(
                example_index=example_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                is_has_answer=example.is_has_answer))

    return features


def load_squad_to_torch_dataset(path_input_data,
                                tokenizer,
                                max_seq_length=500,
                                max_query_length=64,
                                batch_size=20,
                                is_training=True):
    examples = read_squad_example_from_file(path_input_data)
    # print(examples[0])
    shuffle(examples)
    # print(examples[0])

    features = convert_examples_to_features(examples, tokenizer, max_seq_length, max_query_length)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    if is_training:
        all_answerable = torch.tensor([f.is_has_answer for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_answerable)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)

    # cls index using for mark [CLS] token is start or end of pair sentence
    # and p_mask is not mask like mask use for represent the input, not padding, and mask is like opposite
    # we can pass it
    # all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    # all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

    return dataset, train_dataloader


if __name__ == '__main__':

    class Args:
        name_model = "bert-base-multilingual-cased"
        bert_model = '../resources/cache_bert_cased'
        max_seq_length = 500
        predict_batch_size = 20
        batch_size = 20
        n_best_size = 20
        max_answer_length = 30
        do_lower_case = False
        max_query_length = 64
        no_cuda = True
        seed = 42
        THRESH_HOLD = 0.95


    args = Args()

    tokenizer = BertTokenizer.from_pretrained(args.name_model, cache_dir=args.bert_model, do_lower_case=args.do_lower_case)
    path_input_data = "../dataset/sample_pair_sequence.csv"
    load_squad_to_torch_dataset(path_input_data,
                                tokenizer,
                                args.max_seq_length,
                                args.max_query_length,
                                args.batch_size,
                                is_training=True)

