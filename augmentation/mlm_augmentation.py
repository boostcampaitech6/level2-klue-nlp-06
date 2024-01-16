from transformers import AutoTokenizer, AutoModelForMaskedLM

import pandas as pd

import torch

import json
import random
from tqdm import tqdm
import torch.cuda
import ast

def find_indices_with_term(word_list, term):
    indices = [index for index, word in enumerate(word_list) if term in word]
    return indices

def random_integers_except_given(exclude_list, l):
    available_numbers = [num for num in range(l) if num not in exclude_list]
    
    if len(available_numbers) == 0:
        return False
    
    random_integer = random.sample(available_numbers,1)
    for i in random_integer:
        return i

def main(config):
    
    model = AutoModelForMaskedLM.from_pretrained(config['model_name'],cache_dir='/data/ephemeral/tmp')
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'],cache_dir='/data/ephemeral/tmp')
    
    model.to(device)
    
    
    #train aug / val aug 선택
    if config['train_aug'] == True:
        print("train augment 진행!")
        df = pd.read_csv(config['data_path']['train_path'])
        threshold = config['threshold']['train']
    else:
        print("val augment 진행!")
        df = pd.read_csv(config['data_path']['val_path'])
        threshold = config['threshold']['val']

    #threshold에 따른 augmentation label 설정
    label_counts = dict(df['label'].value_counts())
        
    labels_to_augment = [key for key, value in label_counts.items() if value < threshold]
    labels_not_to_augment = [key for key, value in label_counts.items() if value >= threshold]
     
    #no augmentation label dataframe화   
    not_to_augment_df = df[df['label'].isin(labels_not_to_augment)]
    
    list_augmented_df = []
    
    #augment할 label별로 q를 설정하여 문장별로 q번 data augment
    for label in tqdm(labels_to_augment):
    
        num_label = label_counts[label]
        
        if threshold%num_label == 0 :
            q = threshold//num_label -1
        else:
            q = threshold//num_label 
    
        new_index_ids = []
        new_sentences = []
        new_subject_entities = []
        new_object_entities = []
        new_labels = []
        new_sources = []

        for i in range(len(df[df['label']==label])):
            data = df[df['label']==label].iloc[i]
            
            for i in range(q):
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

                modified_subject_word_one = "#" + subject_word_one + "#"
                modified_object_word_one = "#" + object_word_one + "#"
                
                sentence = sentence.replace(subject_word,modified_subject_word_one)
                sentence = sentence.replace(object_word,modified_object_word_one)
                
                for _ in range(2):
                    sentence_split = sentence.split()
                    indices_subject_word_one = find_indices_with_term(sentence_split, modified_subject_word_one)
                    indices_object_word_one = find_indices_with_term(sentence_split, modified_object_word_one)

                    indices_no_include = indices_subject_word_one+indices_object_word_one
                    
                    mask_index = random_integers_except_given(indices_no_include, len(sentence_split))
                    
                    if mask_index == False:
                        continue
                    
                    sentence_split[mask_index] = "[MASK]"
                    
                    sentence = " ".join(sentence_split)
                    
                    tokens = tokenizer(sentence, return_tensors="pt")
                    tokens = {key: value.to('cuda') for key, value in tokens.items()}

                    mask_index = torch.where(tokens["input_ids"] == tokenizer.mask_token_id)

                    with torch.no_grad():
                        outputs = model(**tokens)
                        predictions = outputs.logits[:, mask_index[1], :].argmax(dim=-1)

                    predicted_token = tokenizer.convert_ids_to_tokens(predictions.item())
                    
                    predicted_token = predicted_token.strip()
                    
                    sentence = sentence.replace("[MASK]", predicted_token)
                    
                    
                for _ in range(config['masking_num']):
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
                
                    sentence = sentence.replace("[MASK]", predicted_token)
                
                
                subject_start_idx = sentence.find(modified_subject_word_one)
                object_start_idx = sentence.find(modified_object_word_one)
                
                sentence = sentence.replace(modified_subject_word_one, subject_word)
                sentence = sentence.replace(modified_object_word_one, object_word)

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
    
        list_augmented_df.append(new_data)
    
        
    integrated_data = pd.concat(list_augmented_df, ignore_index=True)
    
    
    #데이터 전처리 
    characters_to_remove = ast.literal_eval(config['remove'])
    for character in characters_to_remove:
        integrated_data['sentence'] = integrated_data['sentence'].str.replace(character,'')
    
    total_data = pd.concat([integrated_data, not_to_augment_df], ignore_index=True )
    
    if config['train_aug'] == True:
        total_data.to_csv('./dataset/'+config['model_name'].split("/")[-1]+'-'+str(config['threshold'])+'-' + 'train.csv', index=False)
    else:
        total_data.to_csv('./dataset/'+config['model_name'].split("/")[-1]+'-'+str(config['threshold'])+'-' + 'dev.csv', index=False)



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
