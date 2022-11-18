import json
import re
from janome.tokenizer import Tokenizer
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
                else:
                    print('登録済み', line)
                if line.startswith('_') and line.endswith('_'):
                    line = line.replace('_', '')
                    if line not in vocab_map:
                        new_vocab[line] = line
    print('新しく追加する語彙数', len(new_vocab))
    return list(new_vocab.keys())

# Japanese from しほまる


t = Tokenizer()

pDigit = re.compile('^[0-9]+$')
pAlpha = re.compile('^[A-Za-z]+$')
pHira = re.compile('[ぁ-ん]')


def containsHira(w):
    return bool(re.search(pHira, w))


def transform(w, s):
    if re.search(pDigit, w):
        return w, '数字'
    if re.search(pAlpha, w):
        return w, '英字'
    return w, s


def janome2(s):
    ws = []
    ss = []
    if s.startswith('▁'):
        s = s[1:]
    for token in t.tokenize(s):
        # 字句と品詞で別のリストを返す
        w, s = transform(token.surface, token.part_of_speech.split(',')[0])
        ws.append(w)
        ss.append(s)
    return ws, ' '.join(ss)


IKI = set(['連体詞 助詞', '助詞 助詞', '動詞 助動詞', '動詞 助動詞 助動詞', '動詞 動詞', '助動詞 助動詞', '動詞 動詞 助動詞', '名詞 助動詞', '名詞 助動詞 助動詞',
          '名詞 名詞', '名詞 名詞 名詞', '名詞 名詞 名詞 名詞', '名詞', '動詞', '副詞', '形容詞', '助詞', '接続詞', '連体詞', '助動詞', '感動詞', 'フィラー', '接頭詞'])

# pDigi = re.compile('[0-9]')


# def containsDigit(w):
#     return bool(re.search(pDigi, w))

def get_trim(ws, vocab_map):
    for w in ws:
        if w not in vocab_map:
            return w
    return None


def remove_vocab(vocab_map, KESU_map, SKIP):
    before = len(KESU_map)
    trimed = {}
    for token in vocab_map.keys():
        if token in SKIP:
            continue
        if containsHira(token):
            ws, pos = janome2(token)
            idx = vocab_map[token]
            if pos not in IKI:
                trim = get_trim(ws, vocab_map)
                if trim and trim not in trimed:
                    KESU_map[token] = trim
                    trimed[trim] = trim
                else:
                    KESU_map[token] = f'<empty_{idx}>'
    print('余分な日本語語彙', len(KESU_map)-before, 'トリム数', len(trimed))

    def remove_gomi(chars, prespace=True):
        nonlocal vocab_map, KESU_map, SKIP
        cc = 0
        for token, idx in vocab_map.items():
            if token in KESU_map or token in SKIP:
                continue
            for c in chars:
                if c in token:
                    if c == token or c in SKIP or (prespace and token == f'▁{c}'):
                        continue
                    if token not in KESU_map:
                        KESU_map[token] = f'<empty_{idx}>'
                        cc += 1
        return cc

    print('全角ゴミ', remove_gomi("。、．・〜（）”“【】「」『』［］｛｝♪～〖〗"))
    print('半角ゴミ', remove_gomi("'+-()[]{}!#$%&=~|`;:,.@?<>\'\"\\'*/_"))
    print('記号ゴミ', remove_gomi("°±¤©＋–×÷£€¢¬●′‚·¶«"))
    print('数字ゴミ', remove_gomi("0123456789"))
    return trimed


# NUM = set(f'{i}' for i in range(10, 512))
# _NUM = set(f'▁{i}' for i in range(10, 512))


def replace_vocab(files, tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    new_model = f'local'
    tokenizer.special_tokens_map_file = "special_tokens_map.json"
    tokenizer.save_pretrained(new_model)
    print('新しいモデルの保存先', new_model)
    m = model.ModelProto()
    m.ParseFromString(open(f"{new_model}/spiece.model", 'rb').read())
    # There are some reserved places for speical tokens
    vocab_map = {}
    for id, piece in enumerate(m.pieces):
        if piece.type == 1:
            token = piece.piece
            vocab_map[token] = id
    print('全語彙数', len(vocab_map))
    new_vocab = read_new_vocab(files, vocab_map)

    KESU_map = {}
    trimed = remove_vocab(vocab_map, KESU_map, set(new_vocab))
    print('消去可能な語彙数', len(KESU_map))
    new_vocab2 = [t for t in new_vocab[::-1] if t not in trimed]

    with open(f'{new_model}/removed.jsonl', 'w') as w:
        for token, newtoken in KESU_map.items():
            idx = vocab_map[token]
            if len(new_vocab2) != 0:
                newtoken = new_vocab2.pop()
            m.pieces[idx].piece = newtoken
            d = {'in': newtoken, 'out': token, 'idx': idx}
            print(json.dumps(d, ensure_ascii=False), file=w)
    if len(m.pieces) > 250000:
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
        tt = tokenizer.encode(v)
        if len(tt) > 3:
            print(v, tt)
    for v in ['<nl><nl>', '<123> <100> <1>']:
        print(v, tokenizer.encode(v))


def main():
    hparams = setup()
    replace_vocab(hparams.files, hparams.tokenizer_path)


if __name__ == '__main__':
    main()
