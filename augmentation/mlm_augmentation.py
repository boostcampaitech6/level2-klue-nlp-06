from transformers import AutoTokenizer, AutoModelForMaskedLM

import pandas as pd

import torch

import json
import random
from tqdm import tqdm
import torch.cuda

def find_word_indices(sentence, word):
    indices = []
    start_index = 0
    while True:
        index = sentence.find(word, start_index)
        if index == -1:
            break
        indices.append((index, index + len(word) - 1))
        start_index = index + len(word) - 1
    return indices


def main(config):
    
    model = AutoModelForMaskedLM.from_pretrained(config['model_name'],cache_dir='/data/ephemeral/tmp')
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'],cache_dir='/data/ephemeral/tmp')
    
    model.to(device)
    
    
    if config['train_aug'] == True:
        print("train augment 진행!")
        df = pd.read_csv(config['data_path']['train_path'])
        threshold = config['threshold']['train']
    else:
        print("val augment 진행!")
        df = pd.read_csv(config['data_path']['val_path'])
        threshold = config['threshold']['val']

    labels = df['label'].unique().tolist()
    label_counts = dict(df['label'].value_counts())
        
    labels_to_augment = [key for key, value in label_counts.items() if value < threshold]
    labels_not_to_augment = [key for key, value in label_counts.items() if value >= threshold]
        
    not_to_augment_df = df[df['label'].isin(labels_not_to_augment)]
    
    
    list_augmented_df = []
    
    
    for label in labels_to_augment:
    
        num_label = label_counts[label]
        
        if threshold%num_label == 0 :
            q = threshold//num_label -1
        else:
            q = 900//num_label 
    
        new_index_ids = []
        new_sentences = []
        new_subject_entities = []
        new_object_entities = []
        new_labels = []
        new_sources = []
    
        for i in tqdm(range(len(df[label]))):
            data = df.iloc[i]
            
            index_id = data["id"]
            sentence = data["sentence"]
            subject_entity = data["subject_entity"]
            object_entity = data["object_entity"]
            label = data["label"]
            source = data["source"]

            #entity에 해당하는 단어 사이에 mask 토큰이 생성되지 않게 띄어쓰기를 없앤 뒤에 mask 토큰을 추가하고 복구
            subject_word = eval(subject_entity)['word']
            object_word = eval(object_entity)['word']

            subject_word_one = subject_word.replace(' ', '')
            object_word_one = object_word.replace(' ', '')

            sentence = sentence.replace(subject_word,subject_word_one)
            sentence = sentence.replace(object_word,object_word_one)
            
            subject_start_idx = find_word_indices(sentence, subject_word_one)[0][0]
            object_start_idx = find_word_indices(sentence, object_word_one)[0][0]
            
            for i in range(q):
                sentence_split = sentence.split()
                random_loc = random.randint(0,len(sentence_split)-1)
                sentence_split.insert(random_loc, "[MASK]")
                sentence = " ".join(sentence_split)
                
                tokens = tokenizer(sentence, return_tensors="pt")
                tokens = {key: value.to('cuda') for key, value in tokens.items()}

                mask_index = torch.where(tokens["input_ids"] == tokenizer.mask_token_id)

                with torch.no_grad():
                    outputs = model(**tokens)
                    predictions = outputs.logits[:, mask_index[1], :].argmax(dim=-1)

                predicted_token = tokenizer.convert_ids_to_tokens(predictions.item())
                
                predicted_token = predicted_token.strip()
                
                token_len = len(predicted_token)
                sentence = sentence.replace("[MASK]", predicted_token)
            
                if sentence[subject_start_idx:subject_start_idx+len(subject_word_one)] != subject_word_one:
                    subject_start_idx += (token_len + 1)
                
                if sentence[object_start_idx:object_start_idx+len(object_word_one)] != object_word_one:
                    object_start_idx += (token_len + 1)
            
            
            sentence = sentence.replace(subject_word_one, subject_word)
            sentence = sentence.replace(object_word_one, object_word)

            if subject_start_idx < object_start_idx:
                object_start_idx += ( len(subject_word) - len(subject_word_one) )
            else:
                subject_start_idx += ( len(object_word) - len(object_word_one) ) 
                
            
            subject_end_idx = subject_start_idx + len(subject_word) -1
            object_end_idx = object_start_idx + len(object_word) -1
            
            new_sentence = sentence 
            subj = eval(subject_entity)["word"]
            obj = eval(object_entity)["word"]
            sub_type = eval(object_entity)["type"]
            obj_type = eval(object_entity)["type"]
            
            new_subj_entity = str(
                {
                    "word": subj,
                    "start_idx": subject_start_idx,
                    "end_idx": subject_end_idx,
                    "type": sub_type,
                }
            )
            new_obj_entity = str(
                {
                    "word": obj,
                    "start_idx": object_start_idx,
                    "end_idx": object_end_idx,
                    "type": obj_type,
                }
            )
            
            ### 증강 시키고, 이전의 데이터도 포함하여 데이터셋을 구축해야한다.
            # 증강 이전의 데이터와, 증강된 데이터는 train/dev 중에서 같은 곳에 존재해야 한다.
            
            
            
            new_index_ids.append(torch.tensor(index_id, dtype=torch.long).to('cuda'))
            new_sentences.append(new_sentence)
            new_subject_entities.append(new_subj_entity)
            new_object_entities.append(new_obj_entity)
            new_labels.append(label)
            new_sources.append(source)

    
    
        new_index_ids = [index_id.item() for index_id in new_index_ids]          
        new_data = pd.DataFrame(
            {
                "id": new_index_ids,
                "sentence": new_sentences,
                "subject_entity": new_subject_entities,
                "object_entity": new_object_entities,
                "label": new_labels,
                "souce": new_sources,
            }
        )
    
    #   new_data = new_data.drop_duplicates()

        list_augmented_df.append(new_data)
    
        
    integrated_data = pd.concat(list_augmented_df, ignore_index=True)
    
    total_data = pd.concat(integrated_data, not_to_augment_df, ignore_index=True )
    
    if config['train_aug'] == True:
        total_data.to_csv('./dataset/'+config['model_name'].split("/")[-1]+'-'+str(config['augment_target'])+'-'+str(config['masking_num'])+'-' + 'train.csv', index=False)
    else:
        total_data.to_csv('./dataset/'+config['model_name'].split("/")[-1]+'-'+str(config['augment_target'])+'-'+str(config['masking_num'])+'-' + 'dev.csv', index=False)

if __name__ == '__main__':
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("gpu 사용중입니다")
    else:
        device = torch.device("cpu")
        print("cpu 사용중입니다.")
    
    with open(f'./configs/mlm_augmentation_config.json', 'r') as f:
        config = json.load(f)

    main(config=config)
