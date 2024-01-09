import pickle as pickle
import os
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
import ast

import transformers
from transformers import AutoTokenizer, AutoConfig
import pytorch_lightning as pl


class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=256)

    def load_data(self, data):
        """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
        # remain original dataset as-is
        out_dataset = data.copy()
        # {'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'} -> '비틀즈'
        out_dataset['subject_entity'] = out_dataset['subject_entity'].apply(lambda x: ast.literal_eval(x)) # turn string formatted like dict into real dict
        out_dataset['object_entity'] = out_dataset['object_entity'].apply(lambda x: ast.literal_eval(x))

        # Entity special token 의 종류를 다양하게 시도해 볼수 있습니다. 
        out_dataset['subject_entity'] = out_dataset['subject_entity'].apply(lambda x: x['word'])
        out_dataset['object_entity'] = out_dataset['object_entity'].apply(lambda x: x['word'])

        return out_dataset

    def tokenizing(self, dataset):
        """ tokenizer에 따라 sentence를 tokenizing 합니다."""
        concat_entity = (dataset['subject_entity'] + '[SEP]' + dataset['object_entity']).values.tolist()

        # [CLS]concat_entity[SEP]sentence[SEP]
        # [CLS]비틀즈[SEP]조지 해리슨[SEP]〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.[SEP]
        tokenized_sentences = self.tokenizer(
            concat_entity,
            list(dataset['sentence']),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True,
        )
        
        return tokenized_sentences

    def label_to_num(self, label: pd.Series):
        """ 문자 label을 숫자로 변환합니다. """
        with open('./utils/dict_label_to_num.pkl', 'rb') as f:
          dict_label_to_num = pickle.load(f)
        num_label = label.map(lambda x: dict_label_to_num.get(x, 0)).values.tolist()
        
        return num_label

    def preprocessing(self, data):
        """ 텍스트 데이터를 전처리합니다. """
        dataset = self.load_data(data)
        inputs = self.tokenizing(dataset)

        targets = self.label_to_num(dataset['label'])

        return inputs, targets

    def setup(self, stage='fit'):
        """ train, validation, inference를 진행합니다. """
        if stage == 'fit':
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            train_inputs, train_targets = self.preprocessing(train_data)

            val_inputs, val_targets = self.preprocessing(val_data)

            self.train_dataset = RE_Dataset(train_inputs, train_targets)
            self.val_dataset = RE_Dataset(val_inputs, val_targets)
        else:
            test_data = pd.read_csv(self.dev_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = RE_Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = RE_Dataset(predict_inputs, predict_targets)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)