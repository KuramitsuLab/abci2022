import argparse
from sentencepiece import sentencepiece_model_pb2 as model

from transformers import AutoTokenizer


def setup():
    parser = argparse.ArgumentParser(description='tokenizer max_length')
    parser.add_argument('files', type=str, nargs='+', help='jsonl files')
    parser.add_argument('--tokenizer_path', default='google/mt5-small')
    hparams = parser.parse_args()  # hparams になる
    return hparams


def read_new_vocab(files):
    new_vocab = {}
    for file in files:
        with open(file) as f:
            for line in f.readlines():
                line = line.rstrip()
                if line == '':
                    continue
                if line not in new_vocab:
                    new_vocab[line] = line
    print('新しい語彙数', len(new_vocab))
    return list(new_vocab.keys())


def remove_gomi_char(vocab_map, c, prespace, removed, new_vocab):
    for token, id in vocab_map.items():
        if c in token:
            if token in new_vocab:
                #print(f'/{token}/', id)
                continue
            if c == token or (prespace and token == f'▁{c}'):
                #print(f'/{token}/', id)
                continue
            removed.add(id)


def remove_unused(vocab_map, new_vocab=set()):
    removed = set()
    for c in '。、．〜（）”“【】「」『』［］｛｝':
        remove_gomi_char(vocab_map, c, False, removed, new_vocab)
    print('全角ゴミ', len(removed))
    for c in '+-()[]{}!#$%&=~|`;:,.@?<>\'\"\\':
        remove_gomi_char(vocab_map, c, True, removed, new_vocab)
    print('半角ゴミ', len(removed))
    for c in '0123456789':
        remove_gomi_char(vocab_map, c, True, removed, new_vocab)
    print('数値ゴミ', len(removed))
    return removed


def replace_vocab(files, tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=False)
    new_model = f'tokenizer-only'
    tokenizer.save_pretrained(new_model)
    print('新しいモデル', new_model)
    m = model.ModelProto()
    m.ParseFromString(open(f"{new_model}/spiece.model", 'rb').read())
    # There are some reserved places for speical tokens
    vocab_map = {}
    for id, piece in enumerate(m.pieces):
        if piece.type == 1:
            vocab_map[piece.piece] = id
    print('語彙数', len(vocab_map))
    new_vocab = read_new_vocab(files)
    removed = remove_unused(vocab_map, set(new_vocab))
    removed = list(sorted(removed, reverse=True))
    with open(f'{new_model}/removed.txt', 'w') as w:
        for new_token in new_vocab:
            if len(removed) == 0:
                break
            idx = removed.pop()
            print(idx, new_token, m.pieces[idx].piece, file=w)
            m.pieces[idx].piece = new_token
    for i, piece in enumerate(m.pieces[250000:], 250000):
        if 'extra_id' in piece.piece:
            piece.piece = piece.piece[1:]
    with open(f"{new_model}/spiece.model", 'wb') as f:
        f.write(m.SerializeToString())


def main():
    hparams = setup()
    replace_vocab(hparams.files, hparams.tokenizer_path)


if __name__ == '__main__':
    main()
