import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from transformers import RobertaModel
import torch
from torch import nn

import pytorch_lightning as pl

from metric import *

class EntityMarkerModel(pl.LightningModule):

    def __init__(self, model_name, lr, tokenizer):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        self.model_config = AutoConfig.from_pretrained(self.model_name)
        self.model_config.num_labels = 30

        # 모델
        # classifier 따로 부착하기
        self.model = RobertaModel.from_pretrained(self.model_name, config=self.model_config)
        # self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.model_config)
        
        # 스페셜 토큰 추가에 따른 토큰 임베딩 조절
        self.model.resize_token_embeddings(len(tokenizer))

        # 토큰 임베딩에 따른 resize layer
        print('### token embeddings : ', self.model.get_input_embeddings())

        # classifier 추가
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.model_config.hidden_size * 3, self.model_config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.model_config.hidden_size * 2, self.model_config.hidden_size),
            nn.GELU(),
            nn.Linear(self.model_config.hidden_size, 30)
        ) # yapf: disable
        self.loss_func = torch.nn.CrossEntropyLoss()

    # Override base_model -> override 안돼서 바꾼 버전..
    def forward(self, input_ids=None, ss=None, os=None, **kwargs):
        input_ids = input_ids.squeeze()
        outputs = self.model(input_ids)

        seq_output = outputs[0]
        pooled_output = outputs[1]

        idx = torch.arange(input_ids.size(0)).to(input_ids.device)

        # subject, object start 임베딩 값 뽑아내기
        ss_emb = seq_output[idx, ss]
        os_emb = seq_output[idx, os]
        h = torch.cat((pooled_output, ss_emb, os_emb), dim=-1)

        logits = self.classifier(h)

        return logits

    def training_step(self, batch, batch_idx):
        x, y, ss, os = batch

        logits = self(x, ss=ss, os=os) # (bs, 30)
        loss = self.loss_func(logits, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, ss, os = batch

        logits = self(x, ss=ss, os=os)
        loss = self.loss_func(logits, y)
        self.log("val_loss", loss)
        
        metrics_dict = compute_metrics(logits,y) 

        for metric_name, metric_value in metrics_dict.items():
            self.log(f"val {metric_name}", metric_value)

        return loss

    def test_step(self, batch, batch_idx):
        x, y, ss, os = batch

        logits = self(x, ss=ss, os=os)
        metrics_dict = compute_metrics(logits,y) 

        for metric_name, metric_value in metrics_dict.items():
            self.log(f"test {metric_name}", metric_value)

    def predict_step(self, batch, batch_idx):
        x, y, ss, os = batch
        logits = self(x, ss=ss, os=os)


        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer