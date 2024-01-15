from transformers import AutoTokenizer, AutoModelForMaskedLM

import pandas as pd

import torch

import json
import random
from tqdm import tqdm

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
    
    train_df = pd.read_csv(config['data_path']['train_path'])
    val_df = pd.read_csv(config['data_path']['val_path'])
    total_df = pd.concat([train_df, val_df], ignore_index=True)
    
    label_counts = total_df['label'].value_counts()
    
    labels_to_augment = label_counts[label_counts <= config['augment_target']].index.to_list()
    target_df = total_df[total_df['label'].isin(labels_to_augment)]
    target_df = target_df.reset_index(drop=True)
    
    
    model = AutoModelForMaskedLM.from_pretrained(config['model_name'],cache_dir='/data/ephemeral/tmp')
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'],cache_dir='/data/ephemeral/tmp')
    
    new_index_ids = []
    new_sentences = []
    new_subject_entities = []
    new_object_entities = []
    new_labels = []
    new_sources = []
    
    for i in tqdm(range(len(target_df))):
        data = target_df.iloc[i]
        
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
        
        for i in range(config["masking_num"]):
            sentence_split = sentence.split()
            random_loc = random.randint(0,len(sentence_split)-1)
            sentence_split.insert(random_loc, "[MASK]")
            sentence = " ".join(sentence_split)
            
            tokens = tokenizer(sentence, return_tensors="pt")

            mask_index = torch.where(tokens["input_ids"] == tokenizer.mask_token_id)

            with torch.no_grad():
                outputs = model(**tokens)
                predictions = outputs.logits[:, mask_index[1], :].argmax(dim=-1)

            predicted_token = tokenizer.convert_ids_to_tokens(predictions.item())
            sentence = sentence.replace("[MASK]", predicted_token)
        

        sentence = sentence.replace(subject_word_one, subject_word)
        sentence = sentence.replace(object_word_one, object_word)

        new_sentence = sentence 
        subj = eval(subject_entity)["word"]
        obj = eval(object_entity)["word"]
        sub_type = eval(object_entity)["type"]
        obj_type = eval(object_entity)["type"]
        
        subj_indices = find_word_indices(new_sentence, subj)
        obj_indices = find_word_indices(new_sentence, obj)
        
        for sub_start_idx, sub_end_idx in subj_indices:
            for obj_start_idx, obj_end_idx in obj_indices:
                new_subj_entity = str(
                    {
                        "word": subj,
                        "start_idx": sub_start_idx,
                        "end_idx": sub_end_idx,
                        "type": sub_type,
                    }
                )
                new_obj_entity = str(
                    {
                        "word": obj,
                        "start_idx": obj_start_idx,
                        "end_idx": obj_end_idx,
                        "type": obj_type,
                    }
                )
                new_index_ids.append(index_id)
                new_sentences.append(new_sentence)
                new_subject_entities.append(new_subj_entity)
                new_object_entities.append(new_obj_entity)
                new_labels.append(label)
                new_sources.append(source)
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
    
    new_data.to_csv('./dataset/'+config['model_name'].split("/")[-1]+'-'+str(config['augment_target'])+'-'+str(config['masking_num'])+'.csv')

if __name__ == '__main__':
    with open(f'./configs/mlm_augmentation_config.json', 'r') as f:
        config = json.load(f)

    main(config=config)
