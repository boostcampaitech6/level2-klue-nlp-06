from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from typing_dataloader import Dataloader

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

from auto_train import set_seed

# deepspeed 딕셔너리 형태로 적재되기 때문에 base model import 필요
import typing_dataloader
from models import typing_model
from utils.utils import *

MODEL = typing_model
DATALOADER = typing_dataloader

# import dataloader
# from models import base_model
# from utils.utils import *

# MODEL = base_model
# DATALOADER = dataloader


def inference(model, tokenized_sent, device, batch_size=16):
    """
      test dataset을 DataLoader로 만들어 준 후,
      batch_size로 나눠 model이 예측 합니다.
    """
    dataloader = DataLoader(tokenized_sent, batch_size, shuffle=False)
    model.eval()
    output_pred = []
    output_prob = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.inference_mode():
            outputs = model(input_ids=data['input_ids'].to(device), 
                            attention_mask=data['attention_mask'].to(device), 
                            token_type_ids=data['token_type_ids'].to(device))

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

    if config['trainer']['deepspeed']:
        CHKPOINT_PATH = get_latest_deepspeed_checkpoint(full_model_name=config['arch']['model_name'],
                                                        model_save_dir="./best_model/",
                                                        return_path=True)
        MODEL_PATH = CHKPOINT_PATH.parent
        checkpoint = torch.load(CHKPOINT_PATH)
        model = getattr(MODEL, config['arch']['model_class'])(config['arch']['model_name'], config['trainer']['learning_rate'])
        model.load_state_dict(checkpoint)

    else:
        MODEL_PATH = get_latest_version(full_model_name=config['arch']['model_name'], 
                                        model_save_dir="./best_model/", 
                                        return_path=True)
        model = torch.load(MODEL_PATH)
      
    print(f"\nCurrently inferencing {MODEL_PATH.stem}...\n")
    model.to(device)

    ## load test datset
    dataloader=getattr(DATALOADER, config['arch']['dataloader_class'])(config['arch']['model_name'], config['trainer']['batch_size'], config['trainer']['shuffle'], 
                                                                       config['path']['train_path'], config['path']['dev_path'], config['path']['test_path'],config['path']['predict_path'])
    
    # dataset 생성
    dataloader.setup(stage='inference')
    
    if config['trainer']['val_mode']:
        pred_dataset = dataloader.test_dataset
    else:
        pred_dataset = dataloader.predict_dataset

    ## predict answer
    pred_answer, output_prob = inference(model,pred_dataset, device, config['trainer']['infer_batch_size']) # model에서 class 추론
    
    ## make csv file with predicted answer
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    if config['trainer']['val_mode']:
        output = pd.DataFrame({'id':pd.Series(range(len(pred_answer))),'pred_label':pred_answer,'probs':output_prob,'label':dataloader.test_dataset[:]['labels'].tolist()})
        save_path = "./prediction/" + "val" + "_" + MODEL_PATH.stem + "_" + "submission.csv"
        output.to_csv(save_path, index=False)
    else:
       pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
       output = pd.DataFrame({'id':pd.Series(range(len(pred_answer))),'pred_label':pred_answer,'probs':output_prob,})
       save_path = "./prediction/" + MODEL_PATH.stem + "_" + "submission.csv"
       output.to_csv(save_path, index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    #### 필수!! ##############################################
    print(f"Result saved in {save_path}")
    print('---- Finish! ----')


if __name__ == '__main__':

    selected_config = 'auto_config.json'

    with open(f'./configs/{selected_config}', 'r') as f:
        config = json.load(f)

    main(config=config)
