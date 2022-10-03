import argparse
import random
import json
import numpy as np
import logging
from logging import INFO, DEBUG, NOTSET
from logging import StreamHandler, FileHandler, Formatter
import click
import dataclasses

import torch
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelSummary
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup
)

from jsonl import JSONL

# global hyperparameter


def setup_hyperparameters():
    USE_GPU = torch.cuda.is_available()
    # ハイパーパラメータの読み込み  何も書かなければ、デフォルト値 default 
    # python3 finetune.py --batch_size 64
    parser = argparse.ArgumentParser(description='train script')
    parser.add_argument('files', type=str, nargs='+', help='jsonl files')
    parser.add_argument('--model_path', default='google/mt5-small')
    parser.add_argument('--tokenizer_path', default=None)
    parser.add_argument('--output_path', default='model')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--source_max_length', type=int, default=None)
    parser.add_argument('--target_max_length', type=int, default=None)
    parser.add_argument('--max_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=0) # 自動
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp_16', type=bool, default=False)
    parser.add_argument('--n_gpus', type=int, default=1 if USE_GPU else 0)

    hparams = parser.parse_args()  # hparams になる
    # デフォルトがNoneのときは
    if hparams.tokenizer_path is None:
        hparams.tokenizer_path = hparams.model_path
    if hparams.source_max_length is None:
        hparams.source_max_length = hparams.max_length
    if hparams.target_max_length is None:
        hparams.target_max_length = hparams.max_length

    # 訓練パラメータの設定
    train_params = dict(
        accumulate_grad_batches=hparams.gradient_accumulation_steps,
        gpus=hparams.n_gpus,
        max_epochs=hparams.max_epochs,
        precision=16 if hparams.fp_16 else 32,
        # amp_level='O1',
        gradient_clip_val=hparams.max_grad_norm,
        # checkpoint_callback=checkpoint_callback,
         # batch_size の自動調整,  hparams.batch_size が上書きされる
        auto_scale_batch_size="binsearch" if hparams.batch_size <= 0 else None,
    )
    return hparams, train_params


def set_seed(seed):  # 乱数シードの設定
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Tokenizer
tokenizer = None


def encode_t5(src, tgt, source_max_length=256, target_max_length=256):
    inputs = tokenizer.batch_encode_plus(
        [src],
        max_length=source_max_length,
        truncation=True,
        pad_to_max_length=True,
        padding="max_length", return_tensors="pt")
    targets = tokenizer.batch_encode_plus(
        [tgt],
        max_length=target_max_length,
        truncation=True,
        pad_to_max_length=True,
        padding="max_length", return_tensors="pt")
    source_ids = inputs["input_ids"].squeeze()
    source_mask = inputs["attention_mask"].squeeze()
    target_ids = targets["input_ids"].squeeze()
    target_mask = targets["attention_mask"].squeeze()
    return {
        "source_ids": source_ids.to(dtype=torch.long),
        "source_mask": source_mask.to(dtype=torch.long),
        "target_ids": target_ids.to(dtype=torch.long),
        "target_mask": target_mask.to(dtype=torch.long),
    }

# FineTuner


class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(hparams.model_path)
        print('pretrained_model', self.model.config)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        """順伝搬"""
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

    def _step(self, batch):
        """ロス計算"""
        labels = batch["target_ids"]
        # All labels set to -100 are ignored (masked),
        # the loss is only computed for labels in [0, ..., config.vocab_size]
        labels[labels[:, :] == tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch['target_mask'],
            labels=labels
        )
        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        """訓練ステップ処理"""
        loss = self._step(batch)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """バリデーションステップ処理"""
        loss = self._step(batch)
        self.log("val_loss", loss, prog_bar=False)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        """バリデーション完了処理"""
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss, prog_bar=False)

    def test_step(self, batch, batch_idx):
        """テストステップ処理"""
        loss = self._step(batch)
        self.log("test_loss", loss, prog_bar=False)
        return {"test_loss": loss}

    def test_epoch_end(self, outputs):
        """テスト完了処理"""
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        self.log("test_loss", avg_loss, prog_bar=False)

    def configure_optimizers(self):
        """オプティマイザーとスケジューラーを作成する"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.learning_rate,
                          eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.t_total
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def setup(self, stage=None):
        """初期設定（データセットの読み込み）"""
        if stage == 'fit' or stage is None:
            # trainデータのパスを指定
            train_dataset = JSONL(
                self.hparams, suffix='_train.', encode=encode_t5)
            self.train_dataset = train_dataset

            # devデータのパスを指定
            val_dataset = JSONL(
                self.hparams, suffix='_valid.', encode=encode_t5)
            self.val_dataset = val_dataset

            self.t_total = (
                (len(train_dataset) //
                 (self.hparams.batch_size * max(1, self.hparams.n_gpus)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.max_epochs)
            )
            print('t_total', self.t_total)

    def train_dataloader(self):
        """訓練データローダーを作成する"""
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          drop_last=True, shuffle=True,
                          num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        """バリデーションデータローダーを作成する"""
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)


def main_train(hparams, train_params):
    set_seed(hparams.seed) # 乱数を初期化
    model = T5FineTuner(hparams)
    trainer = pl.Trainer(**train_params)
    if hparams.batch_size < 1:
        trainer.tune(model)
    if hparams.max_epochs > 0:
        trainer.fit(model)
        # 最終エポックのモデルを保存 output_path に保存します
        tokenizer.save_pretrained(hparams.output_path)
        model.model.save_pretrained(hparams.output_path)


def main_test(hparams):
    set_seed(hparams.seed)
    test_dataset = JSONL(hparams, suffix='_test.')
    if len(test_dataset) == 0:
        return
    test_loader = DataLoader(
        test_dataset,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers)
    # 事前学習済みモデルの読み込み
    model = AutoModelForSeq2SeqLM.from_pretrained(hparams.output_path)
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)
    model.eval()

    inputs = []
    outputs = []
    preds = []
    for batch in test_loader:
        input_ids = batch['source_ids'].to(DEVICE)
        input_mask = batch['source_mask'].to(DEVICE)
        outs = model.generate(
            input_ids=input_ids,
            attention_mask=input_mask,
            max_length=hparams.target_max_length,
            return_dict_in_generate=True,
            output_scores=True)
        dec = [tokenizer.decode(ids, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
               for ids in outs.sequences]
        preds.extend(dec)
        # conf = [s.cpu().item() for s in torch.exp(outs.sequences_scores)]
        inputs.extend([tokenizer.decode(ids, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
                  for ids in batch["source_ids"]])
        outputs.extend([tokenizer.decode(ids, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
                  for ids in batch["target_ids"]])
    # JSONLに保存します。
    with open(f'{hparams.output_dir}/result.jsonl', 'w') as w:
        for ins, out, pred in zip(inputs, outputs, preds):
            line = json.dumps({"in": ins, "out": out, "pred": pred})
            print(line, file=w)



def main():
    global tokenizer # グローバル変数
    hparams, train_params = setup_hyperparameters()
    print('hparams:', hparams)
    print('train_params:', train_params)
    tokenizer = AutoTokenizer.from_pretrained(
        hparams.tokenizer_path, use_fast=False)
    print('tokenizer:', tokenizer)
    main_train(hparams, train_params)
    main_test(hparams)


if __name__ == '__main__': # わかります
    main()
