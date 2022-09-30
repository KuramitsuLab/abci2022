import json
import numpy as np
import logging
from logging import INFO, DEBUG, NOTSET
from logging import StreamHandler, FileHandler, Formatter

import torch
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelSummary
from transformers import (
    MT5ForConditionalGeneration, T5ForConditionalGeneration,
    AutoConfig, AutoModel, AutoTokenizer,
    get_linear_schedule_with_warmup
)

# GPU利用有無
USE_GPU = torch.cuda.is_available()
N_GPU = torch.cuda.device_count()

# Tokenizer 
tokenizer = None

def transform_nop(src, tgt, source_max_length=256, target_max_length=256):
    return (src, tgt)

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

class JSONLDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

def JSONL(files, test=False,
        source='in', target='out', 
        source_max_length=256, target_max_length=256,
        transform=transform_nop):
    dataset=[]
    for file in files:
        logging.info(f'loading {file}')
        if file.endswith('.jsonl'):
            with open(file) as f:
                for c, line in enumerate(f.readlines()):
                    data = json.loads(line)
                    src, tgt = transform(data[source], data[target])
                    dataset.append(encode_t5(src, tgt, source_max_length, target_max_length))
    return JSONLDataset(dataset)

class JSONLDataModule(pl.LightningDataModule):
    def __init__(self, files, batch_size=32, num_workers=8):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        pass
        #torchvision.datasets.MNIST(self.hparams.root_dir, download=True)

    def setup(self, stage):
        if stage == 'fit' or stage is None:
            self.train_dataset = JSONL(
                self.hparams.files, 
                train=True, transform=self.transform
            )

        if stage == 'test' or stage is None:
            self.test_dataset = JSONL(
                self.hparams.files, train=False, transform=self.transform
            )

            # self.t_total = (
            #     (len(self.train_dataset) //
            #      (self.hparams.batch_size * max(1, self.hparams.n_gpu)))
            #     // self.hparams.gradient_accumulation_steps
            #     * float(self.hparams.max_epochs)
            # )


    def train_dataloader(self):
        """訓練データローダーを作成する"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            drop_last=True, shuffle=True,
            num_workers=self.hparams.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers
        )

    # @property
    # def transform(self):
    #     return T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])


class T5FineTuner(pl.LightningModule):
    def __init__(self, model_path):
        super().__init__()
        self.save_hyperparameters()
        self.model = MT5ForConditionalGeneration.from_pretrained(self.hparams.model_path)
        print(self.model.config)

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
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100
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
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        """訓練完了処理"""
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss", loss, prog_bar=self.hparams.prog_bar)
        if not self.hparams.prog_bar:
            print(
                f'Epoch {self.current_epoch} train_loss {loss} train_PPL {math.exp(loss)}')

    def validation_step(self, batch, batch_idx):
        """バリデーションステップ処理"""
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        # """バリデーション完了処理"""
        #print(self.epoch_, outputs)
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss, prog_bar=self.hparams.prog_bar)
        if not self.hparams.prog_bar:
            print(
                f'Epoch {self.current_epoch} val_loss {avg_loss} val_PPL {math.exp(avg_loss)}')
        # self.dataset.split()

    # def test_step(self, batch, batch_idx):
    #     """テストステップ処理"""
    #     loss = self._step(batch)
    #     self.log("test_loss", loss)
    #     return {"test_loss": loss}

    # def test_epoch_end(self, outputs):
    #     """テスト完了処理"""
    #     avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
    #     self.log("test_loss", avg_loss, prog_bar=True)

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
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.t_total
        )
        return [optimizer], [{
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }]

def make_generate(model, tokenizer):
    def greedy_search(s: str, max_length=128) -> str:
        input_ids = tokenizer.encode_plus(
            s,
            add_special_tokens=True,
            max_length=max_length,
            padding="do_not_pad",
            truncation=True,
            return_tensors='pt').input_ids.to(model.device)
        greedy_output = model.generate(input_ids, max_length=max_length)
        return tokenizer.decode(greedy_output[0], skip_special_tokens=True)
    return greedy_search


def _main(model_path, 
    tokenizer_path=None, output_path=None,
    max_epochs=4, batch_size=32):

    if tokenizer_path is None:
        tokenizer_path = model_path # モデルと同じにする
    if output_path is None:
        output_path = f'{model_path}_tuned'

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    hparams = object()
    hparams.batch_size = batch_size
    hparams.prog_bar=False
    hparams.weight_decay
    hparams.learning_rate
    hparams.adam_epsilon
    hparams.warm_steps

    lit = MT5FineTuner(hparams, model)

    train_params = dict(
        enable_model_summary=True,
        accumulate_grad_batches=hparams.gradient_accumulation_steps,
        gpus=hparams.n_gpu,
        max_epochs=hparams.max_epochs,
        # early_stop_callback=False,
        precision=16 if hparams.fp_16 else 32,
        # amp_level=hparams.opt_level,
        gradient_clip_val=hparams.max_grad_norm,
        #    checkpoint_callback=checkpoint_callback,
        # callbacks=[LoggingCallback()],
        callbacks=[
            EarlyStopping(monitor="val_loss"),
            ModelSummary(max_depth=-1)
        ],
        # turn off automatic checkpointing
        enable_checkpointing=True,
        enable_progress_bar=hparams.progress_bar,
        # run batch size scaling, result overrides hparams.batch_size
        auto_scale_batch_size="binsearch" if hparams.batch_size <= 2 else None,
        # run learning rate finder, results override hparams.learning_rate
        # auto_lr_find=True,
        devices="auto", accelerator="auto",
        limit_train_batches=1.0 if hparams.limit_batches == -1 else hparams.limit_batches,
        limit_val_batches=1.0 if hparams.limit_batches == -1 else hparams.limit_batches//4,
    )

    if max_epochs > 0:
        trainer = pl.Trainer(**train_params)
        trainer.tune(net)
        print(f'Start training: max {max_epochs} epochs')
        trainer.fit(lit, dm)
        # 最終エポックのモデルを保存
        tokenizer = model.tokenizer
        model = model.model
        print('saving pretrained ... ', output_path)
        tokenizer.save_pretrained(output_path)
        model.save_pretrained(output_path)

    # DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model.to(DEVICE)
    # train_dataset, valid_dataset = load_TrainTestDataSet(hparams)
    # print('testing ... ', model.device)
    # generate = make_generate(model, tokenizer)
    # valid_dataset.test_and_save(
    #     generate, f'{hparams.output_dir}/result_test.tsv', max=1000)


if __name__ == '__main__':
    _main()
