import os
import json
import sys

from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, DataProcessor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))


class InputExample(object):
    def __init__(self, uid, toks, labels = None):
        self.toks = toks
        self.labels = labels
        self.uid = uid

class InputFeatures(object):
    def __init__(self, eid, input_ids, label_ids, attention_mask) -> None:
        self.input_ids = input_ids
        self.label_ids = label_ids
        self.eid = eid
        self.attention_mask = attention_mask

def convert_example_to_features(example: InputExample, max_seq_length: int, tokenizer:BertTokenizerFast) -> InputFeatures:
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample as ids.
    """
    # 对于单个语句使用tokenizer
    input = tokenizer(example.toks, padding="max_length", return_tensors="pt", max_length=max_seq_length, truncation=True, is_split_into_words=True, return_length=True)
    input_ids = input["input_ids"].squeeze()
    attention_mask = input["attention_mask"].squeeze()


    # pad up to the sequence length
    # while len(input_ids) < max_seq_length:
    #     input_ids.append(0)

    # if example.uid == 0:
    #     print("*** Example ***")
    #     print("uid: %s" % example.uid)
    #     print("tokens: %s" % " ".join([str(x) for x in example.toks]))
    #     print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    #     print("label: %s" % example.labels)
    
    features = InputFeatures(input_ids=input_ids,
                            eid=example.uid,
                            label_ids=example.labels,
                            attention_mask=attention_mask
                            )
    return features


class MyDataset(Dataset):
    """
    construct My dataset to provide features.

    Args:
        texts: ('List[str]')
        labels: ('List[int]')
        tokenizer: BertTokenzierFast
        max_seq_length: 'int'
    """
    def __init__(self, texts, labels, max_seq_length, tokenizer) -> None:
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        toks = self.tokenizer.tokenize(self.texts[index].lower())
        cur_example = InputExample(uid=index, toks=toks, labels=self.labels[index])
        cur_features = convert_example_to_features(cur_example, self.max_seq_length, self.tokenizer)
        cur_tensors = (
            torch.LongTensor(cur_features.input_ids),
            torch.tensor(cur_features.label_ids),
            cur_features.attention_mask
        )
        return cur_tensors



class Processor(DataProcessor):
    def __init__(self, config) -> None:
        super().__init__()
        self.train_path = config.train_path
        self.dev_path = config.dev_path
        self.test_path = config.test_path
        self.tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_tokenizer_path)
        self.labels = config.num_classes
        self.config = config

    def read_data(self, filename):
        with open(filename, "r") as f:
            raw_data = json.load(f)
        text, labels = [], []
        label_name = ["homophonic", "homographic"]
        for i, name in enumerate(label_name):
            samples = raw_data[name]
            for sample in samples:
                text.append(sample['content'])
                labels.append(i)
        return text, labels
        # print(filename)
    
    def get_data(self, mode="train"):
        data, labels = self.read_data(eval("self.%s_path" % mode))
        return data, labels

    def create_dataloader(self, texts, labels, batch_size, shuffle=True):
        """
        params text: [seq_nums, seq_len]
        params labels: []
        """
        # inputs = self.tokenizer(
        #     text,
        #     padding=True,
        #     return_tensors="pt",
        #     max_length=512,
        #     truncation=True
        # )

        dataset = MyDataset(texts, labels, self.config.max_seq_length, self.tokenizer)
        dataloader = DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            )
        
        return dataloader

    def collate_fn():
        pass




if __name__ == "__main__":
    pass