import glob
import argparse
import random
import json
import numpy as np
# import logging
# from logging import INFO, DEBUG, NOTSET
# from logging import StreamHandler, FileHandler, Formatter

import torch
# from torch.utils.data import Dataset
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


def find_latest_checkpoints(checkpoint_dir):
    ckpts = sorted(glob.glob(checkpoint_dir+"/*.ckpt"))
    if len(ckpts) == 0:
        return None
    else:
        return ckpts[-1]


def setup_hyperparameters():
    USE_GPU = torch.cuda.is_available()
    # ハイパーパラメータの読み込み  何も書かなければ、デフォルト値 default
    # python3 finetune.py --batch_size 64
    parser = argparse.ArgumentParser(description='train script')
    parser.add_argument('files', type=str, nargs='+', help='jsonl files')
    parser.add_argument('--model_path', default='google/mt5-small')
    parser.add_argument('--tokenizer_path', default=None)
    parser.add_argument('--checkpoint_path', default=None)
    parser.add_argument('--output_path', default='model')
    parser.add_argument('--tested_file', default='tested.jsonl')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--source_max_length', type=int, default=None)
    parser.add_argument('--target_max_length', type=int, default=None)
    parser.add_argument('--max_epochs', type=int, default=4)
    parser.add_argument('--max_time', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)  # 自動
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument('--n_gpus', type=int, default=1 if USE_GPU else 0)
    # https://note.nkmk.me/python-argparse-bool/
    parser.add_argument('--auto_batch_size',
                        action='store_true', default=False)
    parser.add_argument('--early_stopping', action='store_true', default=False)
    parser.add_argument('--progress_bar', action='store_true', default=False)
    parser.add_argument('--fast_dev_run', action='store_true', default=False)

    hparams = parser.parse_args()  # hparams になる
    # デフォルトがNoneのときは
    if hparams.tokenizer_path is None:
        hparams.tokenizer_path = hparams.model_path
    if hparams.source_max_length is None:
        hparams.source_max_length = hparams.max_length
    if hparams.target_max_length is None:
        hparams.target_max_length = hparams.max_length
    hparams.test = sum(1 for file in hparams.files if '_test.' in file) > 0

    # 訓練パラメータの設定
    # https://torch.classcat.com/2021/02/22/pytorch-lightning-1-1-notebooks-05-trainer-flags-overview-2/
    train_params = dict(
        enable_progress_bar=hparams.progress_bar,
        fast_dev_run=hparams.fast_dev_run,
        gpus=hparams.n_gpus,
        max_epochs=hparams.max_epochs,
        max_time=hparams.max_time,  # "00:00:15:00"
        gradient_clip_val=hparams.max_grad_norm,
        # k バッチ毎に勾配を蓄積する batch_size * k になる
        accumulate_grad_batches=hparams.gradient_accumulation_steps,
        # batch_size の自動調整,  hparams.batch_size が上書きされる
        auto_scale_batch_size="binsearch" if hparams.auto_batch_size else None,
        precision=hparams.precision,
        #        amp_level='O2' if hparams.precision == 16 else 'O0'
    )
    # EarlyStopping
    callbacks = []
    if hparams.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor="val_loss", patience=3,
            verbose=True,
            mode="min"
        )
        callbacks.append(early_stop_callback)
    if hparams.checkpoint_path:
        # https://blog.shikoan.com/pytorch-lightning-max-time/
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            dirpath=hparams.checkpoint_path,
            filename="epoch{epoch:02d}-{val_loss:.5f}",
            save_top_k=3,
            mode="max"
        )
        callbacks.append(checkpoint_callback)
        resume_ckpt = find_latest_checkpoints(hparams.checkpoint_path)
    if len(callbacks) > 0:
        train_params['callbacks'] = callbacks
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


def encode_t5_test(src, tgt, source_max_length=256, target_max_length=256):
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
        "source": src,
        "target": tgt,
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

    def training_epoch_end(self, outputs):
        """バリデーション完了処理"""
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        ppl = torch.exp(avg_loss)
        self.log("train_loss", avg_loss, prog_bar=True)
        self.log("train_ppl", ppl, prog_bar=False)
        if not self.hparams.progress_bar:
            print(
                f'Epoch {self.current_epoch+1} train_loss {avg_loss} PPL {ppl}')

    def validation_step(self, batch, batch_idx):
        """バリデーションステップ処理"""
        loss = self._step(batch)
        self.log("val_loss", loss, prog_bar=False)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        """バリデーション完了処理"""
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        ppl = torch.exp(avg_loss)
        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_ppl", ppl, prog_bar=False)
        if not self.hparams.progress_bar:
            print(
                f'Epoch {self.current_epoch+1} val_loss {avg_loss} PPL {ppl}')

    def test_step(self, batch, batch_idx):
        """テストステップ処理"""
        # print('test batch', batch_idx, batch)
        outputs = self.model.generate(
            input_ids=batch['source_ids'],
            attention_mask=batch['source_mask'],
            max_length=self.hparams.target_max_length,
            return_dict_in_generate=True,
            output_scores=True)
        decs = [tokenizer.decode(ids, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
                for ids in outputs.sequences]
        tested = [(src, tgt, dec) for src, tgt, dec
                  in zip(batch['source'], batch['target'], decs)]
        #self.log("test_loss", loss, prog_bar=False)
        return {"tested": tested}

    def test_epoch_end(self, outputs):
        """テスト完了処理"""
        with open(self.hparams.tested_file, 'w') as w:
            for x in outputs:
                for ins, out, pred in x["tested"]:
                    line = json.dumps(
                        {"in": ins, "out": out, "pred": pred}, ensure_ascii=False)
                    print(line, file=w)

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
        self.t_total = (
            (len(self.train_dataset) //
                (self.hparams.batch_size * max(1, self.hparams.n_gpus)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.max_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
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
            print('train_dataset:', len(self.train_dataset))

            # validデータのパスを指定
            val_dataset = JSONL(
                self.hparams, suffix='_valid.', encode=encode_t5)
            self.val_dataset = val_dataset
            print('val_dataset:', len(self.val_dataset))

        if stage == 'test' or stage is None:
            self.test_dataset = JSONL(
                self.hparams, suffix='_test.', encode=encode_t5_test)
            print('test_dataset:', len(self.test_dataset))

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

    def test_dataloader(self):
        """バリデーションデータローダーを作成する"""
        return DataLoader(self.test_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)


def main_train(hparams, train_params):
    set_seed(hparams.seed)  # 乱数を初期化
    model = T5FineTuner(hparams)
    trainer = pl.Trainer(**train_params)
    if hparams.auto_batch_size:
        trainer.tune(model)
        print('auto_scale_batch_size', model.hparams.batch_size)
    if hparams.max_epochs > 0:
        trainer.fit(model)
        # 最終エポックのモデルを保存 output_path に保存します
        tokenizer.save_pretrained(hparams.output_path)
        model.model.save_pretrained(hparams.output_path)
    if hparams.test:
        trainer.test(model)


def main():
    global tokenizer  # グローバル変数
    hparams, train_params = setup_hyperparameters()
    print('hparams:', hparams)
    tokenizer = AutoTokenizer.from_pretrained(
        hparams.tokenizer_path, use_fast=False)
    print('tokenizer:', tokenizer)
    print('train_params:', train_params)
    main_train(hparams, train_params)


if __name__ == '__main__':
    main()
