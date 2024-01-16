import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from transformers import RobertaModel
import torch
from torch import nn

import pytorch_lightning as pl

from metric import *
import os
CACHE_DIR = '/data/ephemeral/cache'

class EntityMarkerEndtokenModel(pl.LightningModule):

    def __init__(self, model_name, lr, tokenizer):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        self.model_config = AutoConfig.from_pretrained(self.model_name)
        self.model_config.num_labels = 30

        # cache dir 추가
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.model = RobertaModel.from_pretrained(self.model_name, config=self.model_config, cache_dir=CACHE_DIR)
        
        # 스페셜 토큰 추가에 따른 토큰 임베딩 조절
        self.model.resize_token_embeddings(len(tokenizer))

        # 토큰 임베딩에 따른 resize layer
        print('### token embeddings : ', self.model.get_input_embeddings())

        # classifier 추가
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            # concat 5 vector -> resize
            nn.Linear(self.model_config.hidden_size * 5, self.model_config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.model_config.hidden_size * 2, self.model_config.hidden_size),
            nn.GELU(),
            nn.Linear(self.model_config.hidden_size, 30)
        ) # yapf: disable
        self.loss_func = torch.nn.CrossEntropyLoss()

    # Override base_model -> override 안돼서 바꾼 버전..
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, ss=None, os=None, se=None, oe=None):
        input_ids = input_ids.squeeze()
        attention_mask = attention_mask.squeeze()
        token_type_ids = token_type_ids.squeeze()

        model_inputs = {'input_ids' : input_ids, 'token_type_ids' : token_type_ids, 'attention_mask' : attention_mask}

        outputs = self.model(**model_inputs)
        
        seq_output = outputs[0]
        pooled_output = outputs[1]

        idx = torch.arange(input_ids.size(0)).to(input_ids.device)

        # subject, object start 임베딩 값 뽑아내기
        ss_emb = seq_output[idx, ss]
        os_emb = seq_output[idx, os]
        se_emb = seq_output[idx, se]
        oe_emb = seq_output[idx, oe]

        h = torch.cat((pooled_output, ss_emb, se_emb, os_emb, oe_emb), dim=-1)

        logits = self.classifier(h)

        return logits

    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, y, ss, os, se, oe = batch

        logits = self(input_ids, token_type_ids, attention_mask, ss=ss, os=os, se=se, oe=oe) # (bs, 30)
        loss = self.loss_func(logits, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, y, ss, os, se, oe = batch

        logits = self(input_ids, token_type_ids, attention_mask, ss=ss, os=os, se=se, oe=oe)
        loss = self.loss_func(logits, y)
        self.log("val_loss", loss)
        
        metrics_dict = compute_metrics(logits,y) 

        for metric_name, metric_value in metrics_dict.items():
            self.log(f"val {metric_name}", metric_value)

        return loss

    def test_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, y, ss, os, se, oe = batch

        logits = self(input_ids, token_type_ids, attention_mask, ss=ss, os=os, se=se, oe=oe)
        metrics_dict = compute_metrics(logits,y) 

        for metric_name, metric_value in metrics_dict.items():
            self.log(f"test {metric_name}", metric_value)

    def predict_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, y, ss, os, se, oe = batch
        logits = self(input_ids, token_type_ids, attention_mask, ss=ss, os=os, se=se, oe=oe)

        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer