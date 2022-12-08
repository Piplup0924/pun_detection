import torch
from torch import nn
import torch.nn.functional as F
from transformers import BartForSequenceClassification


class CLS_model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.model = BartForSequenceClassification.from_pretrained(config.pretrained_path, num_labels = config.num_classes)

    def forward(self, input_ids, labels, attention_mask = None):
        """
        Params input_ids: [batch_size, seq_len]
        Params attention_mask: [batch_size, seq_len]

        Return outputs: [batch_size, seq_len, num_labels]
        """
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)

        return outputs

    def return_model_config(self):
        return self.model.config.to_json_string()
    
    def save_model(self, model_path):
        return self.model.save_pretrained(model_path)