import datetime
import sys
import json
import black
from nltk import bleu_score
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import re
from io import BytesIO
from tokenize import tokenize, open
from sumeval.metrics.rouge import RougeCalculator
import Levenshtein
import gs

import warnings
warnings.filterwarnings('ignore')

#前処理
def pretreatment(ref, pred):
    try:
        #置換
        new_ref = ref.replace('<nl>','\n').replace('<tab>','    ')
        new_pred = ref.replace('<nl>','\n').replace('<tab>','    ')

        #black
        new_ref = black.format_str(ref,mode=black.Mode())[:-1]
        new_pred = black.format_str(pred,mode=black.Mode())[:-1]
        return ((new_ref, new_pred))
    except:
        global result_black_ng
        result_black_ng += 1
        return ((ref, pred))

def read_file(file_path, results):
    dataset = []
    global result_black_ng
    with open(file_path) as f:
        for line in f.readlines():
            data = json.loads(line)
            ref = data['out']
            pred = data['pred']
            dataset.append(pretreatment(ref, pred))
    results['BLACK_NG件数'] = result_black_ng
    result_black_pass = ((len(dataset)-result_black_ng)/len(dataset))*100
    results['構文パス率'] = round(result_black_pass,3)
    return dataset

def ExactMatch(dataset, results):
    correct_count = 0
    for ref, pred in dataset:
        if ref == pred:
            correct_count += 1
    no_correct_count = len(dataset)-correct_count
    correct_answer_rate = correct_count/len(dataset)*100
    results['全体件数'] = len(dataset)
    results['正答件数'] = correct_count
    results['誤答件数'] = no_correct_count
    results['正答率'] = round(correct_answer_rate,3)

def BLEU(dataset,results):
    pattern = re.compile(r'[\(, .\+\-\)]')

    def tokenize_pycode(code):
        try:
            ss=[]
            tokens = tokenize(BytesIO(code.encode('utf-8')).readline)
            for toknum, tokval, _, _, _ in tokens:
                if toknum != 62 and tokval != '' and tokval != 'utf-8':
                    ss.append(tokval)
            return ss
        except:
            return pattern.split(code)

    references = []
    predictions = []
    for ref ,pred in dataset:
        references.append([tokenize_pycode(ref)])
        predictions.append(tokenize_pycode(pred))
    result_bleu = corpus_bleu(references,predictions)*100
    results['BLEU'] = round(result_bleu,3)

def  CONALA_BLEU(dataset, results):
    smoother = SmoothingFunction()
    sum_b2 = 0
    sum_b4 = 0

    def tokenize_for_bleu_eval(code):
        code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
        code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
        code = re.sub(r'\s+', ' ', code)
        code = code.replace('"', '`')
        code = code.replace('\'', '`')
        tokens = [t for t in code.split(' ') if t]
        return tokens

    for line in dataset:
        py=line[0].strip()
        pred=line[1].strip()
        py = [tokenize_for_bleu_eval(py)]
        pred = tokenize_for_bleu_eval(pred)
        sum_b2 += bleu_score.sentence_bleu(py, pred, smoothing_function=smoother.method2)
        sum_b4 += bleu_score.sentence_bleu(py, pred, smoothing_function=smoother.method4)
    bleu2 = sum_b2 / len(dataset)*100
    bleu4 = sum_b4 / len(dataset)*100
    results['-smooth2'] = round(bleu2,3)
    results['-smooth4'] = round(bleu4,3)

def ROUGE_L(dataset,results):
    rouge = RougeCalculator(lang='ja')
    sum_rouge_score=0
    for line in dataset:
        ref = line[0]
        pred = line[1]
        rouge_score = rouge.rouge_l(summary=pred,references=ref)
        sum_rouge_score += rouge_score
    result_rouge_score = sum_rouge_score/len(dataset)*100
    results['ROUGE-L'] = round(result_rouge_score,3)

def Levenstein(dataset,results):
    sum_levenstein = 0
    for ref ,pred in dataset:
        sum_levenstein += Levenshtein.ratio(ref,pred)
    result_leven = sum_levenstein/len(dataset)*100
    results['Leven'] = round(result_leven,3)


def main():
    results = {'日時':'','ファイル名':'','BLACK_NG件数':'','構文パス率':'',
    '全体件数':'','正答件数':'','誤答件数':'','正答率':'','BLEU':'','-smooth2':'','-smooth5':'','ROUGE-L':'','Leven':''}

    #日付
    datetime_now = datetime.datetime.now()
    results['日時'] = datetime_now.strftime('%Y年%m月%d日 %H:%M:%S')

    #評価するファイル取得
    file_path = sys.argv[1]
    try:
        file_name = sys.argv[2]
    except:
        file_name = sys.argv[1]
    results['ファイル名'] = file_name

    #評価用のデータセット作成
    dataset = read_file(file_path, results)

    #評価用関数実行
    ExactMatch(dataset, results)
    BLEU(dataset, results)
    CONALA_BLEU(dataset, results)
    ROUGE_L(dataset, results)
    Levenstein(dataset, results)

    #結果出力
    gs_result_list = []
    for key, value in results.items():
        if value=='':
            gs_result_list.append('None')
            pass
        else:
            gs_result_list.append(value)
            print(key,":",value)
    gs.send_gs(gs_result_list)

result_black_ng  = 0
main()