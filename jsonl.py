import json
import logging

from torch.utils.data import Dataset


def transform_nop(src, tgt):
    return (src, tgt)


def encode_nop(src, tgt, source_max_length, target_max_length):
    return (src, tgt)


class JSONLDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def JSONL(hparams, suffix='_train',
          source='in', target='out',
          transform=transform_nop, encode=encode_nop):
    dataset = []
    for file in hparams.files:
        if not file.endswith('.jsonl'):
            continue
        if suffix is None or suffix in file:
            logging.info(f'loading {file}..')
            with open(file) as f:
                for line in f.readlines():
                    try:
                        data = json.loads(line)
                        src, tgt = transform(data[source], data[target])
                        data = encode(src, tgt, hparams.source_max_length,
                                    hparams.target_max_length, data)
                        dataset.append(data)
                    except :
                        print('ERR', line)
    return JSONLDataset(dataset)
