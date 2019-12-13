from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
import torch
import torch.nn.functional as F


class BERTQa(BertPreTrainedModel):
    def __init__(self, config):
        super(BERTQa, self).__init__(config)
        self.device = config.device
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.use_pooler = config.use_pooler
        if self.use_pooler:
            self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        else:
            self.qa_outputs_cat = nn.Linear(config.hidden_size * 4, config.num_labels)

        self.weight_class = config.weight_class

        self.init_weights()

    # just need feed input ids and attention mask not need head mask or end position ...
    def compute(self, input_ids, attention_mask=None, token_type_ids=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pooler_output = outputs[1]              # pooler output using for classification task next sent

        if self.use_pooler:
            final_output = self.dropout(pooler_output)
        else:
            hidden_states = outputs[2][1:]
            hidden_states = torch.stack(hidden_states, 1)

            last_4_hidden_states = hidden_states[:, 8:, :, :]

            last_first_4_hidden_states = last_4_hidden_states[:, :, 0, :]
            last_first_4_hidden_states = last_first_4_hidden_states.contiguous().view(
                last_first_4_hidden_states.shape[0],
                last_first_4_hidden_states.shape[1] *
                last_first_4_hidden_states.shape[2])
            final_output = self.dropout(last_first_4_hidden_states)
        return final_output

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        with torch.no_grad():
            final_output = self.compute(input_ids, attention_mask, token_type_ids)
            if self.use_pooler:
                logits = self.qa_outputs(final_output)
            else:
                logits = self.qa_outputs_cat(final_output)
            return logits

    def loss(self, input_ids, attention_mask, token_type_ids, label):
        target = label

        final_output = self.compute(input_ids, attention_mask, token_type_ids)
        if self.use_pooler:
            logits = self.qa_outputs(final_output)
        else:
            logits = self.qa_outputs_cat(final_output)

        class_weights = torch.FloatTensor(self.weight_class).to(self.device)
        loss = F.cross_entropy(logits, target, weight=class_weights)

        predict_value = torch.max(logits, 1)[1]
        list_predict = predict_value.cpu().numpy().tolist()
        list_target = target.cpu().numpy().tolist()

        return loss, list_predict, list_target


if __name__ == '__main__':
    from transformers.configuration_bert import BertConfig

    config = BertConfig.from_pretrained("bert-base-multilingual-uncased",
                                   cache_dir="../resources/cache_model")
    config = config.to_dict()
    config.update({"weight_class": [1, 1]})
    config = BertConfig.from_dict(config)
    # model = BERTQa.from_pretrained("bert-base-multilingual-uncased",
    #                                cache_dir="../resources/cache_model", config=config)
