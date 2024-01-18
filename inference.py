from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from dataloader import *
from dataloader_endtoken import *

from typing import Dict
import json
import pandas as pd
import torch
import torch.nn.functional as F
import random
import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm
import pytorch_lightning as pl

from utils.seed import set_seed

# deepspeed 딕셔너리 형태로 적재되기 때문에 base model import 필요
from models.base_model import Model
from models.entity_marker_model import EntityMarkerModel
from dataloader_prompt2 import *

def inference(model, tokenized_sent, device, batch_size=16):
  """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
  """
  # self, model_name, representation_style, batch_size, shuffle, train_path, dev_path, test_path, predict_path
  dataloader = DataLoader(tokenized_sent, batch_size, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []

  
  for _, data in enumerate(tqdm(dataloader)):

    with torch.inference_mode():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device)
          )

    logits = outputs # (batch_size, 30)

    prob = F.softmax(logits, dim=-1).detach().cpu().numpy() # shape : (batch_size, 30)
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1) # shape : (batch_size, 1)

    output_pred.append(result) 
    output_prob.append(prob)

  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()


def num_to_label(label):
  """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
  """
  origin_label = []
  with open('./utils/dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)

  for v in label:
    origin_label.append(dict_num_to_label[v])
  
  return origin_label

def main(config: Dict):
    set_seed(config)
    """
        주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    ## load my model
    # deepspeed ckpt load 추가
    '''
    deepspeed ckpt로 inference 진행 시, 반드시 zero_to_fp32.py 실행을 통해 fp16으로 저장되어 있는 weights를 fp32로 변환해주는 과정이 필요함
    -> 실행 후 ckpt 폴더 이름과 동일한 이름으로 bin 파일 생성하기
    python3 zero_to_fp32.py '_'.join(config['arch']['model_name'].split('/') + config['arch']['model_detail'].split()) + '.bin'
    '''
    CHKPOINT_PATH = './best_model/' + '_'.join(config['arch']['model_name'].split('/') + config['arch']['model_detail'].split()) + '.ckpt/' +'_'.join(config['arch']['model_name'].split('/') + config['arch']['model_detail'].split()) + '.bin'
    print('### CHKPOINT_PATH : ', CHKPOINT_PATH)

    # baseline 아닌 경우
    # config 존재하는지 여부
    if config['arch']['representation_style'] != "None":
      checkpoint = torch.load(CHKPOINT_PATH)
      tokenizer = AutoTokenizer.from_pretrained(config['arch']['model_name'], max_length=256)

      ### special tokens 추가 ###
      if config['arch']['representation_style'] == 'klue':
         special_tokens_dict = {'additional_special_tokens': ['<subj>', '</subj>', '<obj>', '</obj>']}
        
      elif config['arch']['representation_style'] == 'matching_the_blank':
        special_tokens_dict = {'additional_special_tokens': ['[E1]', '[/E1]', '[E2]', '</E2>']}

      elif config['arch']['representation_style'] ==  'typed_marker':
        special_tokens_dict = {'additional_special_tokens': ['<s:ORG>', '<s:PER>', '</s:ORG>', '</s:PER>', '<o:PER>', '<o:ORG>', '<o:DAT>', '<o:LOC>', '<o:POH>', '<o:NOH>', '</o:PER>', '</o:ORG>', '</o:DAT>', '</o:LOC>', '</o:POH>', '</o:NOH>']}
      
      elif config['arch']['representation_style'] == "typed_punct_marker":
        special_tokens_dict = {'additional_special_tokens': ['*ORG*', '*PER*', '^PER^', '^ORG^', '^DAT^', '^LOC^','^POH^','^NOH^']}

      # 추가 안되는 경우!
      else:
         special_tokens_dict = {'additional_special_tokens' : []}

      num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

      model = EntityMarkerModel(config['arch']['model_name'], config['trainer']['learning_rate'], tokenizer, config['arch']['loss_func'], config['pooling'], config['concat_num'])

      model.load_state_dict(checkpoint)
      model.to(device)

      dataloader = PromptDataloader(config['arch']['model_name'], config['arch']['representation_style'], 
                                config['trainer']['batch_size'], config['trainer']['shuffle'], 
                                config['path']['train_path'], config['path']['dev_path'], config['path']['test_path'],config['path']['predict_path'])
                                

      trainer = pl.Trainer(accelerator="gpu", devices=1,strategy="deepspeed_stage_2", precision=16)
      output_prob = torch.cat(trainer.predict(model=model, datamodule=dataloader))
      output_prob = F.softmax(output_prob.float(), -1)
      pred_answer = output_prob.argmax(-1).tolist()
      output_prob = output_prob.tolist()


    # pt file load
    else:
        MODEL_PATH = './best_model/'+'_'.join(config['arch']['model_name'].split('/') + config['arch']['model_detail'].split()) + '.pt'# model dir.
        model = torch.load(MODEL_PATH)
        model.to(device)
        # dataset 생성
        dataloader=Dataloader(config['arch']['model_name'], 
                              config['trainer']['batch_size'], config['trainer']['shuffle'], 
                              config['path']['train_path'], config['path']['dev_path'], config['path']['test_path'],config['path']['predict_path'])
  
        dataloader.setup(stage='inference')

        if config['trainer']['val_mode']:
          pred_dataset = dataloader.test_dataset
        else:
          pred_dataset = dataloader.predict_dataset

        # predict answer
        pred_answer, output_prob = inference(model, pred_dataset, device, config['trainer']['infer_batch_size']) # model에서 class 추론
    
    pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
    
    ## make csv file with predicted answer
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    if config['trainer']['val_mode']:
        output = pd.DataFrame({'id':pd.Series(range(len(pred_answer))),'pred_label':pred_answer,'probs':output_prob,'label':dataloader.test_dataset[:]['labels'].tolist()})
        output.to_csv('./prediction/'+'val_'+'_'.join(config['arch']['model_name'].split('/') + config['arch']['model_detail'].split())+'_submission.csv', index=False)
    else:
       output = pd.DataFrame({'id':pd.Series(range(len(pred_answer))),'pred_label':pred_answer,'probs':output_prob,})
       output.to_csv('./prediction/'+'_'.join(config['arch']['model_name'].split('/') + config['arch']['model_detail'].split())+'_submission.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    #### 필수!! ##############################################
    print('---- Finish! ----')


if __name__ == '__main__':

    selected_config = 'pretrained_roberta-large_config.json'

    with open(f'./configs/{selected_config}', 'r') as f:
        config = json.load(f)

    main(config=config)