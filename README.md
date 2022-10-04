# abci2022
ABCI Training Script 2022 Version

## ファインチューニング

__使い方__
```
python3 finetune.py music/music_train.jsonl music/music_valid.jsonl
```


## 字句解析器のチェック

コーパスのTokenizer による字句数をチェックする。
`max_length`を設定するときの参考にする。

```
python3 tokenizer_check.py music/music_train.jsonl 
```

* `--tokenizer_path='google/mt5-small'` Tokenizer を指定する
* `--source_max_length=128`: 最大長を超える入力をダンプする
* `--target_max_length=128`: 最大長を超える出力をダンプする