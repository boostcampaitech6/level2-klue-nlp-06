import transformers
from transformers import AutoModel, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
import torch
from torch import nn

import pytorch_lightning as pl

from metric import *
from utils.loss import *
    

class Model(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        self.model_config = AutoConfig.from_pretrained(self.model_name)
        self.model_config.num_labels = 30

        # 모델
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.model_config, cache_dir="/data/ephemeral/home/cache/")

        self.model.resize_token_embeddings(self.model.get_input_embeddings().num_embeddings + 6)

        # Loss 
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, **x):
        x = self.model(**x)['logits']


        return x

    def training_step(self, batch, batch_idx):
        y = batch.pop("labels").long()
        x = batch

        logits = self(**x) # (bs, 30)
        loss = self.loss_func(logits, y) 
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        y = batch.pop("labels").long()
        x = batch

        logits = self(**x)
        loss = self.loss_func(logits, y)
        self.log("val_loss", loss, sync_dist=True)
        
        metrics_dict = compute_metrics(logits,y) 

        for metric_name, metric_value in metrics_dict.items():
            self.log(f"val {metric_name}", metric_value, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        y = batch.pop("labels").long()
        x = batch

        logits = self(**x)

        metrics_dict = compute_metrics(logits,y) 

        for metric_name, metric_value in metrics_dict.items():
            self.log(f"test {metric_name}", metric_value, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        y = batch.pop("labels")
        x = batch

        logits = self(**x) # (bs, 30)


        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    