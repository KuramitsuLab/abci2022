import argparse
import json
import pandas as pd

from transformers import AutoTokenizer


def setup():
    parser = argparse.ArgumentParser(description='tokenizer max_length')
    parser.add_argument('files', type=str, nargs='+', help='jsonl files')
    parser.add_argument('--tokenizer_path', default='google/mt5-small')
    parser.add_argument('--source_max_length', type=int, default=4096)
    parser.add_argument('--target_max_length', type=int, default=4096)
    hparams = parser.parse_args()  # hparams になる
    return hparams


def main():
    hparams = setup()
    tokenizer = AutoTokenizer.from_pretrained(
        hparams.tokenizer_path, use_fast=False)
    for file in hparams.files:
        in_lens = []
        out_lens = []
        with open(file) as f:
            for line in f.readlines():
                d = json.loads(line)
                ids = tokenizer.encode(d['in'])
                in_lens.append(len(ids))
                if len(ids) > hparams.source_max_length:
                    print(len(ids), ids)
                ids = tokenizer.encode(d['out'])
                out_lens.append(len(ids))
                if len(ids) > hparams.target_max_length:
                    print(len(ids), ids)
        print(file)
        df = pd.DataFrame({'in': in_lens, 'out': out_lens})
        print(df.describe())


if __name__ == '__main__':  # わかります
    main()
