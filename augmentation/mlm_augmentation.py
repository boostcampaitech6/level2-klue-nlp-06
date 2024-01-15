from transformers import AutoTokenizer, AutoModelForMaskedLM

import pandas as pd

import json
import random


def main(config):
    
    train_df = pd.read_csv(config['data_path']['train_path'])
    val_df = pd.read_csv(config['data_path']['val_path'])
    total_df = pd.concat([train_df, val_df], ignore_index=True)
    
    label_counts = total_df['label'].value_counts()
    
    labels_to_augment = label_counts[label_counts <= config['augment_target']].index.to_list()
    target_df = total_df[total_df['label'].isin(labels_to_augment)]
    target_df = target_df.reset_index(drop=True)
    
    
    model = AutoModelForMaskedLM.from_pretrained(config['model_name'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    new_index_ids = []
    new_sentences = []
    new_subject_entities = []
    new_object_entities = []
    new_labels = []
    new_sources = []
    
    for i in range(len(target_df)):
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
        
        sentence_split = sentence.split()
        sentence_word_num = len(sentence_split)
        
        for i in range(4):
            random_loc = random.randint(0,sentence_word_num+i-1)
            sentence_split.insert(random_loc, "[MASK]")
            sentence = " ".join(sentence_split)
            
            tokenized_sentence = tokenizer.encode(sentence)
            logits = model(tokenized_sentence).logits
        
    




if __name__ == '__main__':
    with open(f'./configs/mlm_augmentation_config.json', 'r') as f:
        config = json.load(f)

    main(config=config)
