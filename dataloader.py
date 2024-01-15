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
# from utils.replace_representation_tokens import replace_token

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
        # 여기서 갑자기 keyerror?
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



#### 변경된 dataset & dataloader ###

class EntityDataset(torch.utils.data.Dataset):
  """ entity representation 구성을 위한 class."""
  def __init__(self, input_ids, token_type_ids, attention_mask, labels, ss_arr=None, os_arr=None):
    self.input_ids = input_ids
    self.token_type_ids = token_type_ids
    self.attention_mask = attention_mask
    
    self.labels = labels

    # ss, os 추가를 위해 넣어주기
    self.ss_arr = ss_arr
    self.os_arr = os_arr

  def __getitem__(self, idx):
    if len(self.labels) == 0:
        return (torch.tensor(self.input_ids[idx]), torch.tensor(self.attention_mask[idx]), torch.tensor(self.token_type_ids[idx]),
                torch.tensor([]), torch.tensor(self.ss_arr[idx]), torch.tensor(self.os_arr[idx]))
    
    else:

        return (torch.tensor(self.input_ids[idx]), torch.tensor(self.attention_mask[idx]), torch.tensor(self.token_type_ids[idx]),
                torch.tensor(self.labels[idx]), torch.tensor(self.ss_arr[idx]), torch.tensor(self.os_arr[idx]))


  def __len__(self):
    return len(self.labels)


class EntityDataloader(pl.LightningDataModule):
    def __init__(self, model_name, representation_style, batch_size, shuffle, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.representation_style = representation_style
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
        # 스페셜 토큰이 추가되는 representation 방식의 경우 각 스페셜 토큰을 tokenizer에 add
        # ['baseline', 'klue', 'matching_the_blank', 'punct', 'typed_marker', 'typed_punct_marker']
        print("### representation style : ", self.representation_style)
        print("### special tokens in current tokenizer : ", self.tokenizer.special_tokens_map)
        
        
        if self.representation_style == 'klue':
           special_tokens_dict = {'additional_special_tokens': ['<subj>', '</subj>', '<obj>', '</obj>']}
        
        elif self.representation_style == 'matching_the_blank':
           special_tokens_dict = {'additional_special_tokens': ['[E1]', '[/E1]', '[E2]', '</E2>']}
        

        elif self.representation_style == 'typed_marker':
            # type에 따른 리스트 지정 새로 필요
            # dataset -> type unique 추출해서 모든 조합 붙이기 (in ipynb)
            special_tokens_dict = {'additional_special_tokens': ['<s:ORG>', '<s:PER>', '</s:ORG>', '</s:PER>', '<o:PER>', '<o:ORG>', '<o:DAT>', '<o:LOC>', '<o:POH>', '<o:NOH>', '</o:PER>', '</o:ORG>', '</o:DAT>', '</o:LOC>', '</o:POH>', '</o:NOH>']}

        # special tokens 추가
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        
        print('### We have added', num_added_toks, 'tokens.')
        print("### special tokens in current tokenizer : ", self.tokenizer.special_tokens_map)

        # ValueError
        if self.representation_style not in ['klue', 'matching_the_blank', 'punct', 'typed_marker', 'typed_punct_marker']:
            raise ValueError("Wrong representation style input detected. Please choose one of ['baseline', 'klue', 'matching_the_blank', 'punct', 'typed_marker', 'typed_punct_marker']")
        

    def load_data(self, data):
        """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
        # remain original dataset as-is
        out_dataset = data.copy()
        out_dataset['subject_entity'] = out_dataset['subject_entity'].apply(lambda x: ast.literal_eval(x)) # turn string formatted like dict into real dict
        out_dataset['object_entity'] = out_dataset['object_entity'].apply(lambda x: ast.literal_eval(x))

        return out_dataset

    def tokenizing(self, dataset):
        '''
        tokenizer에 따라 sentence를 tokenizing 합니다.
        ['baseline', 'klue', 'matching_the_blank', 'punct', 'typed_marker', 'typed_punct_marker']

        baseline:
            [CLS]concat_entity[SEP]sentence[SEP]
            [CLS]비틀즈[SEP]조지 해리슨[SEP]〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.[SEP]
        klue:
            [CLS]〈Something〉는 <obj>조지 해리슨</obj>이 쓰고 <subj>비틀즈</subj>가 1969년 앨범 《Abbey Road》에 담은 노래다.[SEP]
        matching_the_blank:
            [CLS]〈Something〉는 [E2]조지 해리슨[/E2]이 쓰고 [E1]비틀즈[/E1]가 1969년 앨범 《Abbey Road》에 담은 노래다.[SEP]
        punct:
            [CLS]〈Something〉는 #조지 해리슨#이 쓰고 @비틀즈@가 1969년 앨범 《Abbey Road》에 담은 노래다.[SEP]
        typed_marker:
            [CLS]〈Something〉는 <o:PER>조지 해리슨</o:PER>이 쓰고 <s:ORG>비틀즈</s:ORG>가 1969년 앨범 《Abbey Road》에 담은 노래다.[SEP]
        typed_punct_marker:
            [CLS]〈Something〉는 #^PER^조지 해리슨#이 쓰고 @*ORG*비틀즈@가 1969년 앨범 《Abbey Road》에 담은 노래다.[SEP]
        '''
        
        # out dataset에 추가하기
        input_ids, attention_mask, token_type_ids, ss_arr, os_arr = [], [], [], [], []
        for _, item in tqdm(dataset.iterrows(), desc='tokenizing', total=len(dataset)):
            x_sentence = item['sentence']

            if self.representation_style == 'klue':
                entity_markers = {'subj_s' : '<subj>', 'subj_e' : '</subj>', 'obj_s' : '<obj>', 'obj_e' : '</obj>'}
            elif self.representation_style == 'matching_the_blank':
                entity_markers = {'subj_s' : '[E1]', 'subj_e' : '</E1>', 'obj_s' : '<E2>', 'obj_e' : '</E2>'}
            elif self.representation_style == 'punct':
                entity_markers = {'subj_s' : '@', 'subj_e' : '@', 'obj_s' : '#', 'obj_e' : '#'}
                    

            subj_s = item['subject_entity']['start_idx']
            subj_e = item['subject_entity']['end_idx']
            obj_s = item['object_entity']['start_idx']
            obj_e = item['object_entity']['end_idx']

            subj_type = item['subject_entity']['type']
            obj_type = item['object_entity']['type']

            subj_word = item['subject_entity']['word']
            obj_word = item['object_entity']['word']

            # entity type 포함
            tmp = []
            if self.representation_style == 'typed_marker':
                # [CLS]〈Something〉는 <o:PER>조지 해리슨</o:PER>이 쓰고 <s:ORG>비틀즈</s:ORG>가 1969년 앨범 《Abbey Road》에 담은 노래다.[SEP]
                if subj_s < obj_s: # subj 가 먼저 나올 때
                    tmp.extend([
                                    x_sentence[:subj_s],
                                    f'<s:{subj_type}>{subj_word}</s:{subj_type}>',
                                    x_sentence[subj_e+1:obj_s], 
                                    f'<o:{obj_type}>{obj_word}</o:{obj_type}>',
                                    x_sentence[obj_e+1:]
                                ])
                    
                elif subj_s > obj_s: # obj 가 먼저 나올 때
                    tmp.extend([
                                    x_sentence[:obj_s],
                                    f'<o:{obj_type}>{obj_word}</o:{obj_type}>',
                                    x_sentence[obj_e+1:subj_s], 
                                    f'<s:{subj_type}>{subj_word}</s:{subj_type}>',
                                    x_sentence[subj_e+1:]
                                ])
                else:
                    raise ValueError("subj-obj overlapped")
                

            elif self.representation_style == 'typed_punct_marker':
                if subj_s < obj_s: # subj 가 먼저 나올 때
                    tmp.extend([
                                x_sentence[:subj_s],
                                f'@*{subj_type}*{subj_word}@',
                                x_sentence[subj_e+1:obj_s],
                                f'#^{obj_type}^{obj_word}#',
                                x_sentence[obj_e+1:]
                            ])
                    
                elif subj_s > obj_s: # obj 가 먼저 나올 때
                    tmp.extend([
                                    x_sentence[:obj_s],
                                    f'#^{obj_type}^{obj_word}#',
                                    x_sentence[obj_e+1:subj_s],
                                    f'@*{subj_type}*{subj_word}@',
                                    x_sentence[subj_e+1:]
                                ])
                else:
                    raise ValueError("subj-obj overlapped")


            # entity type 불포함
            else:
                if subj_s < obj_s: # subj 가 먼저 나올 때
                    tmp.extend([
                            x_sentence[:subj_s],
                            entity_markers['subj_s'] + subj_word + entity_markers['subj_e'],
                            x_sentence[subj_e+1:obj_s],
                            entity_markers['obj_s'] + obj_word + entity_markers['obj_e'],
                            x_sentence[obj_e+1:]
                    ])

                elif subj_s > obj_s: # obj 가 먼저 나올 때
                    tmp.extend([
                            x_sentence[:obj_s],
                            entity_markers['obj_s'] + obj_word + entity_markers['obj_e'],
                            x_sentence[obj_e+1:subj_s],
                            entity_markers['subj_s'] + subj_word + entity_markers['subj_e'],
                            x_sentence[subj_e+1:]
                    ])

                else:
                    raise ValueError("subj-obj overlapped")

            # tokenized sentence, ss, os 반환하기
            # subject 시작 위치 (토큰 단위 계산)
            ss = len(self.tokenizer(tmp[0], add_special_tokens=False)['input_ids']) + 1
            os = ss + len(self.tokenizer(tmp[1], add_special_tokens=False)['input_ids']) + len(self.tokenizer(tmp[2], add_special_tokens=False)['input_ids'])
                
            if subj_s > obj_s:
                ss, os = os, ss
                
            text = "".join(tmp)

            outputs = self.tokenizer(text, 
                                    return_tensors="pt", 
                                    padding="max_length", 
                                    truncation=True, 
                                    max_length=256, 
                                    add_special_tokens=True)

            input_ids.append(outputs['input_ids'])
            token_type_ids.append(outputs['token_type_ids'])
            attention_mask.append(outputs['attention_mask'])
            
            ss_arr.append(ss)
            os_arr.append(os)
        
        
        # return tokenized_sentences, ss_arr, os_arr
        # {input_ids, token_type_ids, attention_mask}, ss_arr, os_arr
        return input_ids, token_type_ids, attention_mask, ss_arr, os_arr


    def label_to_num(self, label: pd.Series):
        """ 문자 label을 숫자로 변환합니다. """
        with open('./utils/dict_label_to_num.pkl', 'rb') as f:
          dict_label_to_num = pickle.load(f)
        num_label = label.map(lambda x: dict_label_to_num.get(x, 0)).values.tolist()
        
        return num_label

    def preprocessing(self, data):
        """ 텍스트 데이터를 전처리합니다. """
        dataset = self.load_data(data)
        input_ids, token_type_ids, attention_mask, ss_arr, os_arr = self.tokenizing(dataset)

        targets = self.label_to_num(dataset['label'])

        return input_ids, token_type_ids, attention_mask, targets, ss_arr, os_arr

    def setup(self, stage='fit'):
        """ train, validation, inference를 진행합니다. """
        if stage == 'fit':
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            train_inputs_ids, train_token_type_ids, train_attention_mask, train_targets, train_ss_arr, train_os_arr = self.preprocessing(train_data)
            val_inputs_ids, val_token_type_ids, val_attention_mask, val_targets, valid_ss_arr, valid_os_arr = self.preprocessing(val_data)

            self.train_dataset = EntityDataset(train_inputs_ids, train_token_type_ids, train_attention_mask, train_targets, train_ss_arr, train_os_arr)
            self.val_dataset = EntityDataset(val_inputs_ids, val_token_type_ids, val_attention_mask, val_targets, valid_ss_arr, valid_os_arr)
        else:
            test_data = pd.read_csv(self.dev_path)

            test_inputs_ids, test_token_type_ids, test_attention_mask, test_targets, test_ss_arr, test_os_arr = self.preprocessing(test_data)
            self.test_dataset = EntityDataset(test_inputs_ids, test_token_type_ids, test_attention_mask, test_targets, test_ss_arr, test_os_arr)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs_ids, predict_token_type_ids, predict_attention_mask, predict_targets, predict_ss_arr, predict_os_arr = self.preprocessing(predict_data)
            self.predict_dataset = EntityDataset(predict_inputs_ids, predict_token_type_ids, predict_attention_mask, predict_targets, predict_ss_arr, predict_os_arr)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)