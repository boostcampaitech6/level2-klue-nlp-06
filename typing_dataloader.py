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

        special_tokens_dict = {
            "additional_special_tokens": [
                "[PER]",
                "[ORG]",
                "[LOC]",
                "[DAT]",
                "[POH]",
                "[NOH]"
            ]
        }

        self.tokenizer.add_special_tokens(special_tokens_dict)

    def load_data(self, data):
        """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
        # remain original dataset as-is
        out_dataset = data.copy()

        # {'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'} -> '비틀즈'
        # 여기서 갑자기 keyerror?
        out_dataset['subject_entity'] = out_dataset['subject_entity'].apply(lambda x: ast.literal_eval(x)) # turn string formatted like dict into real dict
        out_dataset['object_entity'] = out_dataset['object_entity'].apply(lambda x: ast.literal_eval(x))

        se_dict = {f"se_{k}" : [] for k in out_dataset['subject_entity'][0].keys()}
        oe_dict = {f"oe_{k}" : [] for k in out_dataset['object_entity'][0].keys()}

        for _, row in out_dataset[['subject_entity', 'object_entity']].iterrows():
            for k, v in row['subject_entity'].items():
                se_dict[f"se_{k}"].append(v)
            for k, v in row['object_entity'].items():
                oe_dict[f"oe_{k}"].append(v)
        
        for k, v in se_dict.items():
            out_dataset[k] = v
        for k, v in oe_dict.items():
            out_dataset[k] = v
        
        out_dataset['se_type'] = "[" + out_dataset['se_type'] + "]"
        out_dataset['oe_type'] = "[" + out_dataset['oe_type'] + "]"

        out_dataset['marked_se_type'] = "* " + out_dataset['se_type'] + " *"
        out_dataset['marked_oe_type'] = "^ " + out_dataset['oe_type'] + " ^"

        out_dataset['marked_se_word'] = "@ " + out_dataset['marked_se_type'] + " " + out_dataset['se_word'] + " @"
        out_dataset['marked_oe_word'] = "# " + out_dataset['marked_oe_type'] + " " + out_dataset['oe_word'] + " #"

        def full_sentence(row):
            subsitute_sentence = row['sentence'].replace(row['se_word'], row['marked_se_word']).replace(row['oe_word'], row['marked_oe_word'])
            full_sentence = f"{row['marked_se_word']}와 {row['marked_oe_word']} 사이의 관계는 무었인가?" + " " + subsitute_sentence + " " + f"{row['se_word']}와 {row['oe_word']}의 관계는 {row['se_type']}와 {row['oe_type']}의 관계이다."
            # full_sentence = subsitute_sentence + " " + f"이 문장에서 {row['marked_se_word']}와 {row['marked_oe_word']} 사이의 관계는 무었인가?" + " " + f"{row['se_word']}와 {row['oe_word']}의 관계는 {row['se_type']}와 {row['oe_type']}의 관계이다."
            return full_sentence
        
        out_dataset['full_sentence'] = out_dataset.apply(full_sentence, axis=1)
        return out_dataset

    def tokenizing(self, dataset):
        """ tokenizer에 따라 sentence를 tokenizing 합니다."""
        # '〈Something〉는 # ^ [PER] ^ 조지 해리슨이 쓰고 @ * [ORG] * 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다. 이 문장에서 @ * [ORG] * 비틀즈와 # ^ [PER] ^ 조지 해리슨 사이의 관계는 무었인가? 비틀즈와 조지 해리슨의 관계는 [ORG]와 [PER]의 관계이다.'
        # '@ * [ORG] * 비틀즈와 # ^ [PER] ^ 조지 해리슨 사이의 관계는 무었인가? 〈Something〉는 # ^ [PER] ^ 조지 해리슨이 쓰고 @ * [ORG] * 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다. 비틀즈와 조지 해리슨의 관계는 [ORG]와 [PER]의 관계이다.'
        tokenized_sentences = self.tokenizer(
            list(dataset['full_sentence']),
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