import sys
import json
import janome

from janome.tokenizer import Tokenizer
janome = Tokenizer()


def merge(w, w2, pos, pos2):
    if pos == '形容詞':
        if w2 == 'さ':
            return w+w2, '名詞'
    if pos == '名詞':
        if pos2 == '名詞':
            return w+w2, '名詞'
        if pos2 == '動詞':
            return w+w2, '動詞'
        if pos2 == '助動詞':
            return w+w2, '形容詞'
    if pos == '接頭詞' and pos2 == '名詞':
        return w+w2, '名詞'
    if pos == '動詞' and pos2 == '助詞':
        return w+w2, '動詞'
    if pos == '助詞' and pos2 == '助詞':
        return w+w2, '助詞'
    return None


def janome_tkn(s):
    ws = []
    pos = []
    for t in janome.tokenize(s):
        w = t.surface
        po = t.part_of_speech.split(',')[0]
        if len(ws) > 0:
            merged = merge(ws[-1], w, pos[-1], po)
            if merged:
                ws[-1] = merged[0]
                pos[-1] = merged[1]
                continue
        ws.append(w)
        pos.append(po)
    return ws, pos


IMPORTANT_POS = set(['名詞', '動詞', '形容詞', '副詞'])


def mask_jp(s, d='', file=sys.stdout):
    keywords = []
    for w, pos in zip(*janome_tkn(s)):
        if pos in IMPORTANT_POS and len(w) > 1:
            keywords.append(w)
    print(keywords)
    for keyword in keywords:
        p = s.replace(keyword, '')
        p = json.dumps({'key': keyword, 'in': p, 'out': d}, ensure_ascii=False)
        print(p, file=file)


def main():
    mask_jp('建物の屋根上を発電事業者に貸し、初期費用ゼロで太陽光パネルを設置できる「電力購入契約モデル（PPA）」が九州・沖縄で広がってきた。')


if __name__ == '__main__':
    main()
