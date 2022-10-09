import argparse
from sentencepiece import sentencepiece_model_pb2 as model

from transformers import AutoTokenizer


def setup():
    parser = argparse.ArgumentParser(description='tokenizer max_length')
    parser.add_argument('files', type=str, nargs='+', help='jsonl files')
    parser.add_argument('--tokenizer_path', default='google/mt5-small')
    hparams = parser.parse_args()  # hparams になる
    return hparams


def read_new_vocab(files, vocab_map):
    new_vocab = {}
    for file in files:
        with open(file) as f:
            for line in f.readlines():
                if '\t' in line or ' ' in line:
                    line = line.split()[0]
                else:
                    line = line.strip()
                if line == '' or line in vocab_map:
                    continue
                if line not in new_vocab:
                    new_vocab[line] = line
                if line.startswith('_') and line.endswith('_'):
                    line = line.replace('_', '')
                    if line not in vocab_map:
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


def test_vocab(tokenizer_path, new_vocab):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=False)
    for v in new_vocab():
        print(v, tokenizer.encode(v))


NUM = set(f'{i}' for i in range(10, 512))
_NUM = set(f'▁{i}' for i in range(10, 512))


def replace_vocab(files, tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=False)
    new_model = f'tokenizer-only'
    tokenizer.special_tokens_map_file = "special_tokens_map.json"
    tokenizer.save_pretrained(new_model)
    print('新しいモデル', new_model)
    m = model.ModelProto()
    m.ParseFromString(open(f"{new_model}/spiece.model", 'rb').read())
    # There are some reserved places for speical tokens
    vocab_map = {}
    for id, piece in enumerate(m.pieces):
        if piece.type == 1:
            token = piece.piece
            # if token.endswith('>'):
            #     print(token)
            # if token in NUM:
            #     token = f'<{token}>'
            #     m.pieces[id].piece = token
            # if token in _NUM:
            #     token = f'▁<{token[1:]}>'
            #     m.pieces[id].piece = token
            vocab_map[token] = id
    print('全語彙数', len(vocab_map))
    new_vocab = read_new_vocab(files, vocab_map)
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
    test_vocab(new_model, new_vocab)


def test_vocab(tokenizer_path, new_vocab):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=False)
    for v in new_vocab:
        print(v, tokenizer.encode(v))
    for v in ['<nl><nl>', '<123> <100> <1>']:
        print(v, tokenizer.encode(v))


def main():
    hparams = setup()
    replace_vocab(hparams.files, hparams.tokenizer_path)


if __name__ == '__main__':
    main()
