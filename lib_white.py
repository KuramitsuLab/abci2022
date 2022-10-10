import random
import numpy as np
import json
import keyword
import re


def safeid(n):
    return f'_qid_{n}_'


def replace_quote(code, qmap, quote, qsize):
    p = code.find(quote)
    if p != -1:
        s = p-1 if p > 1 and code[p-1] in 'fur' else p
        p2 = code.find(quote, p+qsize)
        # print(f'>{quote}>', p, p+qsize, p2, f'/{code}/')
        if p2 >= p+qsize:
            key = safeid(len(qmap))
            qmap[key] = code[s:p2 +
                             qsize].replace('<”>', '\\"').replace('<’>', "\\'")
            # print('>>>', p2+qsize, f'/{code[p2+qsize:]}/')
            r = replace_quote(code[p2+qsize:], qmap, quote, qsize)
            # print('>>>', p2+qsize, f'/{r}/')
            return code[:s]+key+r
        else:
            key = safeid(len(qmap))
            qmap[key] = code[s:].replace('<”>', '\\"').replace('<’>', "\\'")
            return code[:s]+key+f' #Err {quote}'
    return code


def extract_quote(code):
    code = code.replace("\\\n", '')
    qmap = {}
    code = replace_quote(code, qmap, '"""', 3)
    code = replace_quote(code, qmap, "''''", 3)
    ss = []
    for line in code.splitlines():
        line = line.replace('\\"', '<”>').replace("\\'", '<’>').rstrip('\n')
        line = replace_quote(line, qmap, '"', 1)
        line = replace_quote(line, qmap, "'", 1)
        p = line.find('#')
        if p >= 0:
            key = safeid(len(qmap))
            qmap[key] = line[p:]
            line = line[:p] + ' ' + key
        ss.append(line)
    #print('@@', qmap)
    return '\n'.join(ss), qmap


OPCHARS = '+*/%=!<>&|^~@'


def white_tokenize(text: str) -> list:
    """
    （Python に限らず）ソースコードを字句分割する

    Args:
        text (str): ソースコード

    Returns:
        list 字句リスト
    """
    text = text.replace('<nl>', '\n').replace('<tab>', '\t')
    text, qmap = extract_quote(text)
    text = text.replace('    ', '\t')
    text = re.sub(r'([^A-Za-z0-9_])', r' \1 ', text)
    text = text.replace('\n', '<nl>').replace('\t', ' <tab>')
    # text = re.sub(r'\s+', ' ', text)
    toks = [t for t in text.split(' ') if t != '']
    for i in range(1, len(toks)):
        p, n = toks[i-1][-1], toks[i][0]
        if p == '>' and n == '<':
            pass
        elif p in OPCHARS and n in OPCHARS:
            toks[i] = toks[i-1]+toks[i]
            toks[i-1] = ''
    tokens = [qmap[t] if t in qmap else t for t in toks if t != '']
    return tokens


def white_pack(source_or_tokens, token_map={}):
    if isinstance(source_or_tokens, str):
        tokens = white_tokenize(source_or_tokens)
    else:
        tokens = source_or_tokens
    prev = '@'
    ss = []
    for tok in tokens:
        if tok in token_map:
            tok = token_map[tok]
        head = tok[0]
        if (prev.isalnum() or prev == '_') and (head.isalnum() or head == '_'):
            ss.append(' ')
        ss.append(tok)
        prev = tok[-1]
    return ''.join(ss)


SPACING_PREV = ',=<>+-*/%!|&'
SPACING_HEAD = '+-*/%=<>!|&'


def white_format(code, token_map={}):
    if isinstance(code, str):
        tokens = white_tokenize(code)
    else:
        tokens = code
    prev = '@'
    ss = []
    mapped = {'<nl>': '\n', '<tab>': '    '}
    mapped.update(token_map)
    for tok in tokens:
        if tok in mapped:
            tok = mapped[tok]
        head = tok[0]
        if (prev.isalnum() or prev == '_') and (head.isalnum() or head == '_'):
            ss.append(' ')
        elif head in SPACING_HEAD or prev in SPACING_PREV:
            ss.append(' ')
        ss.append(tok)
        prev = tok[-1]
    return ''.join(ss)
