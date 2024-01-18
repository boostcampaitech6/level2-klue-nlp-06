import pickle as pickle
import os
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
import ast
from pathlib import Path

import transformers
from transformers import AutoTokenizer, AutoConfig, BertTokenizerFast, BertConfig
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
    

class EntityDataloader(pl.LightningDataModule):
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

        self.entity_category = {
            'subject': 1,
            'object': 2,
            'other': 0
        }

        self.tokenizer = BertTokenizerFast.from_pretrained(model_name, max_length=256) # MUST use Fast tokenizer

    def load_data(self, data):
        """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
        # remain original dataset as-is
        out_dataset = data.copy()

        # {'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'} -> 각 key 를 column 으로 분리
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
        out_dataset = out_dataset[['id', 'sentence', 'se_word', 'se_start_idx', 'se_end_idx', 'se_type', 'oe_word', 'oe_start_idx', 'oe_end_idx', 'oe_type', 'label', 'source']]

        return out_dataset

    def tokenizing(self, dataset, dataset_type="train"):
        """ tokenizer에 따라 sentence를 tokenizing 합니다."""
        concat_entity = (dataset['se_word'] + '[SEP]' + dataset['oe_word']).values.tolist()

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

        sep_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        tokenized_sentences['entity_ids'] = self.entity_encoding(dataset, tokenized_sentences, sep_id=sep_id, dataset_type=dataset_type)
        
        return tokenized_sentences
    
    def entity_encoding(self, dataset, tokenized_sentences, sep_id=2, dataset_type="train"):
        """ tokenized 된 결과를 바탕으로 각 token 이 subject 또는 object 에 속하는지 알려주는 entity encoding 을 만듭니다."""
        encoding_dir = Path("./encoding/")
        encoding_dir.mkdir(exist_ok=True)
        encoding_model_name = '_'.join(self.model_name.split('/')) # klue/roberta-large -> klue_roberta-large
        encoding_dataset_name = ""
        if dataset_type == "train": 
            encoding_dataset_name = self.train_path.split('/')[-1].split('.')[0] # ./dataset/without_dup_train.csv -> without_dup_train
        elif dataset_type == "val":
            encoding_dataset_name = self.dev_path.split('/')[-1].split('.')[0]
        elif dataset_type == "test":
            encoding_dataset_name = self.test_path.split('/')[-1].split('.')[0]
        elif dataset_type == "predict":
            encoding_dataset_name = self.predict_path.split('/')[-1].split('.')[0]
        else:
            raise ValueError("dataset_type must be one of train, val, test, predict")
        encoding_file_name = f"{encoding_model_name}+{encoding_dataset_name}.pt" # klue_roberta-large+without_dup_train.pt
        encoding_path = encoding_dir / encoding_file_name

        if encoding_path.exists():
            print(f"\nEncoding for {dataset_type} already exists. Load entity encoding from {encoding_path}...\n")
            entity_encodings = torch.load(encoding_path)
            return torch.tensor(entity_encodings)
        
        print(f"\nCreate entity encoding for {dataset_type} dataset...")
        entity_encodings = []

        # 문장 by 문장, 토큰 by 토큰 O(n^2)
        for i in tqdm(range(tokenized_sentences['input_ids'].shape[0])):
            # [CLS]비틀즈[SEP]조지 해리슨[SEP]〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.[SEP]
            entity_encoding = []
            sep_idices = torch.where(tokenized_sentences['input_ids'][i] == sep_id)[0]
            concat_sep_idx = sep_idices[0] # 비틀즈[SEP]조지 해리슨
            sentence_sep_idx = sep_idices[1] # concat_entity 와 sentence 사이의 [SEP]
            current_row = dataset.iloc[i]
            
            for j in range(tokenized_sentences['input_ids'].shape[1]):
                index_range = tokenized_sentences.token_to_chars(i, j)
                if index_range is None: # 처음에 삽입된 [CLS], 문장 사이와 맨끝에 삽입된 [SEP] -> 자동 삽입 되었으므로 건너뜁니다. 
                    entity_encoding.append(self.entity_category['other'])
                    continue
                # 비틀즈[SEP]조지 해리슨
                if j < sentence_sep_idx:
                    if j < concat_sep_idx:
                        category = self.entity_category['subject']
                    elif j > concat_sep_idx:
                        category = self.entity_category['object']
                    else:
                        category = self.entity_category['other']
                # 〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.
                elif j > sentence_sep_idx:
                    if (index_range.start >= current_row['se_start_idx']) and (index_range.end <= current_row['se_end_idx'] + 1):
                        category = self.entity_category['subject']
                    elif (index_range.start >= current_row['oe_start_idx']) and (index_range.end <= current_row['oe_end_idx'] + 1):
                        category = self.entity_category['object']
                    else: 
                        category = self.entity_category['other']
                # concat_entity 와 sentence 사이의 [SEP]
                else:
                    category = self.entity_category['other']
                entity_encoding.append(category)

            entity_encodings.append(entity_encoding)

        print(f"Save created entity encoding to {encoding_path}...\n")
        torch.save(entity_encodings, encoding_path)
        return torch.tensor(entity_encodings)

    def label_to_num(self, label: pd.Series):
        """ 문자 label을 숫자로 변환합니다. """
        with open('./utils/dict_label_to_num.pkl', 'rb') as f:
          dict_label_to_num = pickle.load(f)
        num_label = label.map(lambda x: dict_label_to_num.get(x, 0)).values.tolist()
        
        return num_label

    def preprocessing(self, data, dataset_type="train"):
        """ 텍스트 데이터를 전처리합니다. """
        dataset = self.load_data(data)
        inputs = self.tokenizing(dataset, dataset_type=dataset_type)

        targets = self.label_to_num(dataset['label'])

        return inputs, targets

    def setup(self, stage='fit'):
        """ train, validation, inference를 진행합니다. """
        if stage == 'fit':
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            train_inputs, train_targets = self.preprocessing(train_data, dataset_type="train")

            val_inputs, val_targets = self.preprocessing(val_data, dataset_type="val")

            self.train_dataset = RE_Dataset(train_inputs, train_targets)
            self.val_dataset = RE_Dataset(val_inputs, val_targets)
        else:
            test_data = pd.read_csv(self.dev_path)
            test_inputs, test_targets = self.preprocessing(test_data, dataset_type="test")
            self.test_dataset = RE_Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data, dataset_type="predict")
            self.predict_dataset = RE_Dataset(predict_inputs, predict_targets)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)
    

    

class EntityBinaryDataloader(pl.LightningDataModule):
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

        self.entity_category = {
            'entity': 1,
            'other': 0
        }

        self.tokenizer = BertTokenizerFast.from_pretrained(model_name, max_length=256) # MUST use Fast tokenizer

    def load_data(self, data):
        """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
        # remain original dataset as-is
        out_dataset = data.copy()

        # {'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'} -> 각 key 를 column 으로 분리
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
        out_dataset = out_dataset[['id', 'sentence', 'se_word', 'se_start_idx', 'se_end_idx', 'se_type', 'oe_word', 'oe_start_idx', 'oe_end_idx', 'oe_type', 'label', 'source']]

        return out_dataset

    def tokenizing(self, dataset, dataset_type="train"):
        """ tokenizer에 따라 sentence를 tokenizing 합니다."""
        concat_entity = (dataset['se_word'] + '[SEP]' + dataset['oe_word']).values.tolist()

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

        sep_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        tokenized_sentences['entity_ids'] = self.entity_encoding(dataset, tokenized_sentences, sep_id=sep_id, dataset_type=dataset_type)
        
        return tokenized_sentences
    
    def entity_encoding(self, dataset, tokenized_sentences, sep_id=2, dataset_type="train"):
        """ tokenized 된 결과를 바탕으로 각 token 이 subject 또는 object 에 속하는지 알려주는 entity encoding 을 만듭니다."""
        encoding_dir = Path("./encoding/")
        encoding_dir.mkdir(exist_ok=True)
        encoding_model_name = '_'.join(self.model_name.split('/')) # klue/roberta-large -> klue_roberta-large
        encoding_dataset_name = ""
        if dataset_type == "train": 
            encoding_dataset_name = self.train_path.split('/')[-1].split('.')[0] # ./dataset/without_dup_train.csv -> without_dup_train
        elif dataset_type == "val":
            encoding_dataset_name = self.dev_path.split('/')[-1].split('.')[0]
        elif dataset_type == "test":
            encoding_dataset_name = self.test_path.split('/')[-1].split('.')[0]
        elif dataset_type == "predict":
            encoding_dataset_name = self.predict_path.split('/')[-1].split('.')[0]
        else:
            raise ValueError("dataset_type must be one of train, val, test, predict")
        encoding_file_name = f"{encoding_model_name}+{encoding_dataset_name}+binary.pt" # klue_roberta-large+without_dup_train.pt
        encoding_path = encoding_dir / encoding_file_name

        if encoding_path.exists():
            print(f"\nBinary encoding for {dataset_type} already exists. Load entity encoding from {encoding_path}...\n")
            entity_encodings = torch.load(encoding_path)
            return torch.tensor(entity_encodings)
        
        print(f"\nCreate binary entity encoding for {dataset_type} dataset...")
        entity_encodings = []

        # 문장 by 문장, 토큰 by 토큰 O(n^2)
        for i in tqdm(range(tokenized_sentences['input_ids'].shape[0])):
            # [CLS]비틀즈[SEP]조지 해리슨[SEP]〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.[SEP]
            entity_encoding = []
            sep_idices = torch.where(tokenized_sentences['input_ids'][i] == sep_id)[0]
            concat_sep_idx = sep_idices[0] # 비틀즈[SEP]조지 해리슨
            sentence_sep_idx = sep_idices[1] # concat_entity 와 sentence 사이의 [SEP]
            current_row = dataset.iloc[i]
            
            for j in range(tokenized_sentences['input_ids'].shape[1]):
                index_range = tokenized_sentences.token_to_chars(i, j)
                if index_range is None: # 처음에 삽입된 [CLS], 문장 사이와 맨끝에 삽입된 [SEP] -> 자동 삽입 되었으므로 건너뜁니다. 
                    entity_encoding.append(self.entity_category['other'])
                    continue
                # 비틀즈[SEP]조지 해리슨
                if j < sentence_sep_idx:
                    if j == concat_sep_idx:
                        category = self.entity_category['other']
                    else:
                        category = self.entity_category['entity']
                # 〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.
                elif j > sentence_sep_idx:
                    if (index_range.start >= current_row['se_start_idx']) and (index_range.end <= current_row['se_end_idx'] + 1):
                        category = self.entity_category['entity']
                    elif (index_range.start >= current_row['oe_start_idx']) and (index_range.end <= current_row['oe_end_idx'] + 1):
                        category = self.entity_category['entity']
                    else: 
                        category = self.entity_category['other']
                # concat_entity 와 sentence 사이의 [SEP]
                else:
                    category = self.entity_category['other']
                entity_encoding.append(category)

            entity_encodings.append(entity_encoding)

        print(f"Save created entity encoding to {encoding_path}...\n")
        torch.save(entity_encodings, encoding_path)
        return torch.tensor(entity_encodings)

    def label_to_num(self, label: pd.Series):
        """ 문자 label을 숫자로 변환합니다. """
        with open('./utils/dict_label_to_num.pkl', 'rb') as f:
          dict_label_to_num = pickle.load(f)
        num_label = label.map(lambda x: dict_label_to_num.get(x, 0)).values.tolist()
        
        return num_label

    def preprocessing(self, data, dataset_type="train"):
        """ 텍스트 데이터를 전처리합니다. """
        dataset = self.load_data(data)
        inputs = self.tokenizing(dataset, dataset_type=dataset_type)

        targets = self.label_to_num(dataset['label'])

        return inputs, targets

    def setup(self, stage='fit'):
        """ train, validation, inference를 진행합니다. """
        if stage == 'fit':
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            train_inputs, train_targets = self.preprocessing(train_data, dataset_type="train")

            val_inputs, val_targets = self.preprocessing(val_data, dataset_type="val")

            self.train_dataset = RE_Dataset(train_inputs, train_targets)
            self.val_dataset = RE_Dataset(val_inputs, val_targets)
        else:
            test_data = pd.read_csv(self.dev_path)
            test_inputs, test_targets = self.preprocessing(test_data, dataset_type="test")
            self.test_dataset = RE_Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data, dataset_type="predict")
            self.predict_dataset = RE_Dataset(predict_inputs, predict_targets)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)
    

    

class EntityTypeDataloader(pl.LightningDataModule):
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

        ### [주의]
        ### entity_model.py 의 EntityRobertaEmbeddings class 에서             
        ### self.entity_embeddings 의 vocab size 를 key 의 갯수인 7 로 수정해주세요.
        self.entity_category = {'other': 0, 'PER': 1, 'ORG': 2, 'LOC': 3, 'DAT': 4, 'POH': 5, 'NOH': 6}

        self.tokenizer = BertTokenizerFast.from_pretrained(model_name, max_length=256) # MUST use Fast tokenizer

    def load_data(self, data):
        """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
        # remain original dataset as-is
        out_dataset = data.copy()

        # {'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'} -> 각 key 를 column 으로 분리
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
        out_dataset = out_dataset[['id', 'sentence', 'se_word', 'se_start_idx', 'se_end_idx', 'se_type', 'oe_word', 'oe_start_idx', 'oe_end_idx', 'oe_type', 'label', 'source']]

        return out_dataset

    def tokenizing(self, dataset, dataset_type="train"):
        """ tokenizer에 따라 sentence를 tokenizing 합니다."""
        concat_entity = (dataset['se_word'] + '[SEP]' + dataset['oe_word']).values.tolist()

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

        sep_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        tokenized_sentences['entity_ids'] = self.entity_encoding(dataset, tokenized_sentences, sep_id=sep_id, dataset_type=dataset_type)
        
        return tokenized_sentences
    
    def entity_encoding(self, dataset, tokenized_sentences, sep_id=2, dataset_type="train"):
        """ tokenized 된 결과를 바탕으로 각 token 이 subject 또는 object 에 속하는지 알려주는 entity encoding 을 만듭니다."""
        encoding_dir = Path("./encoding/")
        encoding_dir.mkdir(exist_ok=True)
        encoding_model_name = '_'.join(self.model_name.split('/')) # klue/roberta-large -> klue_roberta-large
        encoding_dataset_name = ""
        if dataset_type == "train": 
            encoding_dataset_name = self.train_path.split('/')[-1].split('.')[0] # ./dataset/without_dup_train.csv -> without_dup_train
        elif dataset_type == "val":
            encoding_dataset_name = self.dev_path.split('/')[-1].split('.')[0]
        elif dataset_type == "test":
            encoding_dataset_name = self.test_path.split('/')[-1].split('.')[0]
        elif dataset_type == "predict":
            encoding_dataset_name = self.predict_path.split('/')[-1].split('.')[0]
        else:
            raise ValueError("dataset_type must be one of train, val, test, predict")
        encoding_file_name = f"{encoding_model_name}+{encoding_dataset_name}+type.pt" # klue_roberta-large+without_dup_train.pt
        encoding_path = encoding_dir / encoding_file_name

        if encoding_path.exists():
            print(f"\nType encoding for {dataset_type} already exists. Load entity encoding from {encoding_path}...\n")
            entity_encodings = torch.load(encoding_path)
            return torch.tensor(entity_encodings)
        
        print(f"\nCreate type entity encoding for {dataset_type} dataset...")
        entity_encodings = []

        # 문장 by 문장, 토큰 by 토큰 O(n^2)
        for i in tqdm(range(tokenized_sentences['input_ids'].shape[0])):
            # [CLS]비틀즈[SEP]조지 해리슨[SEP]〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.[SEP]
            entity_encoding = []
            sep_idices = torch.where(tokenized_sentences['input_ids'][i] == sep_id)[0]
            concat_sep_idx = sep_idices[0] # 비틀즈[SEP]조지 해리슨
            sentence_sep_idx = sep_idices[1] # concat_entity 와 sentence 사이의 [SEP]
            current_row = dataset.iloc[i]
            
            for j in range(tokenized_sentences['input_ids'].shape[1]):
                index_range = tokenized_sentences.token_to_chars(i, j)
                if index_range is None: # 처음에 삽입된 [CLS], 문장 사이와 맨끝에 삽입된 [SEP] -> 자동 삽입 되었으므로 건너뜁니다. 
                    entity_encoding.append(self.entity_category['other'])
                    continue
                # 비틀즈[SEP]조지 해리슨
                if j < sentence_sep_idx:
                    if j < concat_sep_idx:
                        category = self.entity_category[current_row['se_type']]
                    elif j > concat_sep_idx:
                        category = self.entity_category[current_row['oe_type']]
                    else:
                        category = self.entity_category['other']
                # 〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.
                elif j > sentence_sep_idx:
                    if (index_range.start >= current_row['se_start_idx']) and (index_range.end <= current_row['se_end_idx'] + 1):
                        category = self.entity_category[current_row['se_type']]
                    elif (index_range.start >= current_row['oe_start_idx']) and (index_range.end <= current_row['oe_end_idx'] + 1):
                        category = self.entity_category[current_row['oe_type']]
                    else: 
                        category = self.entity_category['other']
                # concat_entity 와 sentence 사이의 [SEP]
                else:
                    category = self.entity_category['other']
                entity_encoding.append(category)

            entity_encodings.append(entity_encoding)

        print(f"Save created entity encoding to {encoding_path}...\n")
        torch.save(entity_encodings, encoding_path)
        return torch.tensor(entity_encodings)

    def label_to_num(self, label: pd.Series):
        """ 문자 label을 숫자로 변환합니다. """
        with open('./utils/dict_label_to_num.pkl', 'rb') as f:
          dict_label_to_num = pickle.load(f)
        num_label = label.map(lambda x: dict_label_to_num.get(x, 0)).values.tolist()
        
        return num_label

    def preprocessing(self, data, dataset_type="train"):
        """ 텍스트 데이터를 전처리합니다. """
        dataset = self.load_data(data)
        inputs = self.tokenizing(dataset, dataset_type=dataset_type)

        targets = self.label_to_num(dataset['label'])

        return inputs, targets

    def setup(self, stage='fit'):
        """ train, validation, inference를 진행합니다. """
        if stage == 'fit':
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            train_inputs, train_targets = self.preprocessing(train_data, dataset_type="train")

            val_inputs, val_targets = self.preprocessing(val_data, dataset_type="val")

            self.train_dataset = RE_Dataset(train_inputs, train_targets)
            self.val_dataset = RE_Dataset(val_inputs, val_targets)
        else:
            test_data = pd.read_csv(self.dev_path)
            test_inputs, test_targets = self.preprocessing(test_data, dataset_type="test")
            self.test_dataset = RE_Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data, dataset_type="predict")
            self.predict_dataset = RE_Dataset(predict_inputs, predict_targets)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)