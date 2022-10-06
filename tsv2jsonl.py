import json
import csv
import argparse


def transform_nop(s):
    return s


def setup():
    parser = argparse.ArgumentParser(description='train script')
    parser.add_argument('files', type=str, nargs='+', help='jsonl files')
    parser.add_argument('--in', type=int, default=0)
    parser.add_argument('--out', type=int, default=1)
    parser.add_argument('--transform_in', type=str, default=transform_nop)
    parser.add_argument('--transform_out', type=str, default=transform_nop)
    hparams = parser.parse_args()  # hparams になる
    print(hparams)
    return hparams


def load_tsv(file, column=0, target_column=1):
    ss = []
    with open(file) as f:
        for cols in csv.reader(f, delimiter='\t'):
            p = (cols[column], cols[target_column])
            print(p)
            ss.append(p)
    print(f'Loaded {len(ss)}')
    newfile = file.replace('.tsv', '.jsonl')
    return ss, newfile


def store_jsonl(file, ss, transform_in, transform_out):
    with open(file, 'w') as w:
        for ins, ous in ss:
            json_data = {
                'in': transform_in(ins),
                'out': transform_out(ous),
            }
            print(json.dumps(json_data), file=w)


def main():
    hparams = setup()
    for file in hparams.files:
        ss, newfile = load_tsv(file, getattr(hparams, 'in'), hparams.out)
        store_jsonl(newfile, ss, hparams.transform_in, hparams.transform_out)


if __name__ == '__main__':
    main()
