import os
import json
import pandas as pd
from torch.utils.data.dataset import Dataset
import torch

import argparse

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AugmentDataset(Dataset):
    def __init__(self, augment_config, tokenizer):
        
        train_df = pd.read_csv(augment_config['data_path']['train_path'])
        test_df = pd.read_csv(augment_config['data_path']['test_path'])
        total_df = pd.concat([train_df, test_df], ignore_index=True)
        
        self.tokenizer = tokenizer
        
        self.sentence_list = []

        for i in range(len(total_df)):
            sent = total_df.iloc[i, 1]
            self.sentence_list.append(sent)
        
        self.tokenized_sentence = self.tokenizer(
            self.sentence_list,
            return_tensors=augment_config["tokenizers"]["return_tensors"],
            padding=augment_config["tokenizers"]["padding"],
            truncation=augment_config["tokenizers"]["truncation"],
            max_length=augment_config["tokenizers"]["maxlength"],
            add_special_tokens=augment_config["tokenizers"]["add_special_tokens"],
        )

        self.tokenized_sentences = []
        
        for idx in range(len(self.tokenized_sentence["input_ids"])):
            temp = {key: val[idx].to(device) for key, val in self.tokenized_sentence.items()}
            self.tokenized_sentences.append(temp)
            

    def __len__(self):
        return len(self.tokenized_sentences)

    def __getitem__(self, idx):
        return self.tokenized_sentences[idx]

if __name__ == "__main__":
    selected_config = 'augment_config.json'

    with open(f'./configs/{selected_config}', 'r') as f:
        augment_config = json.load(f)

    model = AutoModelForMaskedLM.from_pretrained(augment_config["model_name"])
    
    tokenizer = AutoTokenizer.from_pretrained(augment_config["model_name"])
    dataset = AugmentDataset(
        augment_config = augment_config,
        tokenizer = tokenizer
        )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer = tokenizer,
        mlm = augment_config["DataCollatorForLanguageModeling"]["mlm"],
        mlm_probability = augment_config["DataCollatorForLanguageModeling"]["mlm_probability"],
        )

    training_args = TrainingArguments(
        output_dir = augment_config["training_arguments"]["output_dir"],
        overwrite_output_dir = augment_config["training_arguments"]["overwrite_output_dir"],
        learning_rate = augment_config["training_arguments"]["learning_rate"],
        num_train_epochs = augment_config["training_arguments"]["num_train_epochs"],
        per_device_train_batch_size = augment_config["training_arguments"]["per_gpu_train_batch_size"],
        gradient_accumulation_steps = augment_config["training_arguments"]["gradient_accumulation_steps"],
        save_steps = augment_config["training_arguments"]["save_steps"],
        save_total_limit = augment_config["training_arguments"]["save_total_limit"],
        logging_steps = augment_config["training_arguments"]["logging_steps"],
        weight_decay = augment_config["training_arguments"]["weight_decay"]
        )

    trainer = Trainer(
        model = model.to(device),
        args = training_args,
        data_collator = data_collator,
        train_dataset = dataset,
        )    

    trainer.train(resume_from_checkpoint="./augmentation/output_v1/checkpoint-14000")
    trainer.save_model(augment_config["data_path"]["save_path"])