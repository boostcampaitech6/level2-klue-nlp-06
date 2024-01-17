import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
import torch
from torch import nn

import pytorch_lightning as pl

from metric import *
import os
CACHE_DIR = '/data/ephemeral/cache'

# classifier head 추가..
class BaseClassifier(nn.Module):
    def __init__(self, model_config, concat_num=3):
        super(BaseClassifier, self).__init__()
        self.model_config = model_config

        self.layers = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.model_config.hidden_size * concat_num, self.model_config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.model_config.hidden_size * 2, self.model_config.hidden_size),
            nn.GELU(),
            nn.Linear(self.model_config.hidden_size, 30)
        ) # yapf: disable
        

    def forward(self, x):
        return self.layers(x)

# classifier head 추가..
class PoolingClassifier(nn.Module):
    def __init__(self, model_config):
        super(PoolingClassifier, self).__init__()
        self.model_config = model_config

        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool1d(self.model_config.hidden_size),
            nn.Dropout(p=0.2),
            nn.Linear(self.model_config.hidden_size, self.model_config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.model_config.hidden_size * 2, self.model_config.hidden_size),
            nn.GELU(),
            nn.Linear(self.model_config.hidden_size, 30)
        ) # yapf: disable
        

    def forward(self, x):
        return self.layers(x)


class EntityMarkerModel(pl.LightningModule):
    # pooling, loss function, concat_num 추가
    def __init__(self, model_name, lr, tokenizer, loss_func, pooling, concat_num):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        self.model_config = AutoConfig.from_pretrained(self.model_name)
        self.model_config.num_labels = 30

        # cache dir 추가
        os.makedirs(CACHE_DIR, exist_ok=True)
        # AutoModel로 change
        self.model = AutoModel.from_pretrained(self.model_name, config=self.model_config, cache_dir=CACHE_DIR)
        
        # 스페셜 토큰 추가에 따른 토큰 임베딩 조절
        self.model.resize_token_embeddings(len(tokenizer))

        # 토큰 임베딩에 따른 resize layer
        print('### token embeddings : ', self.model.get_input_embeddings())

        # classifier 추가
        if self.pooling == False:
            self.classifier = BaseClassifier(self.model_config, self.concat_num)
        else:
            self.classifier = PoolingClassifier(self.model_config)

        # Loss 
        if self.loss_func == 'focal':
            self.loss_func = FocalLoss()
        elif self.loss_func == 'CE':
            self.loss_func = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("Wrong loss_func input. Choose one of ['focal', 'CE']")

    # Override base_model -> override 안돼서 바꾼 버전..
    # input_ids -> tokenized_outputs(input_ids, token_type_ids, attention_masks)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, ss=None, os=None, **kwargs):
        input_ids = input_ids.squeeze()
        attention_mask = attention_mask.squeeze()
        token_type_ids = token_type_ids.squeeze()

        model_inputs = {'input_ids' : input_ids, 'token_type_ids' : token_type_ids, 'attention_mask' : attention_mask}

        outputs = self.model(**model_inputs)
        
        seq_output = outputs[0]
        pooled_output = outputs[1]

        idx = torch.arange(input_ids.size(0)).to(input_ids.device)

        # subject, object start & end 임베딩 값 뽑아내기
        ss_emb = seq_output[idx, ss]
        os_emb = seq_output[idx, os]
        se_emb = seq_output[idx, se]
        oe_emb = seq_output[idx, oe]

        # CLS, ss, os
        if self.concat_num == 3:
            h = torch.cat([pooled_output, ss_emb, os_emb], dim=-1)
        elif self.concat_num == 4:
            h = torch.cat([ss_emb, se_emb, os_emb, oe_emb], dim=-1)
        elif self.concat_num == 5:
            h = torch.cat([pooled_output, ss_emb, se_emb, os_emb, oe_emb], dim=-1)
        else:
            raise ValueError("Wrong concat_num input. Choose 3~5")

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