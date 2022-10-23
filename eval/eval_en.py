import datetime
import sys
import json
from nltk.translate.bleu_score import corpus_bleu
import re
from sumeval.metrics.rouge import RougeCalculator
import Levenshtein
import gs

import warnings
warnings.filterwarnings('ignore')

#前処理
def pretreatment(ref, pred):
    try:
        #置換
        new_ref = ref.replace(" ","").replace('<nl>','\n').replace('<tab>','    ')
        new_pred = pred.replace(" ","").replace('<nl>','\n').replace('<tab>','    ')
        return ((new_ref, new_pred))
    except:
        return ((ref, pred))

def read_file(file_path):
    dataset = []
    with open(file_path) as f:
        for line in f.readlines():
            data = json.loads(line)
            ref = data['out']
            pred = data['pred']
            dataset.append(pretreatment(ref, pred))
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
    results['正答率'] = round(correct_answer_rate,5)

def BLEU(dataset,results):
    pattern = re.compile(r'[\(, .\+\-\)]')

    references = []
    predictions = []
    for ref ,pred in dataset:
        references.append([ref])
        predictions.append(pred)
    result_bleu = corpus_bleu(references,predictions)*100
    results['BLEU'] = round(result_bleu,3)

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
    leven = sum_levenstein/len(dataset)*100
    results['leven'] = round(leven,3)


def main():
    results = {'日時':'','ファイル名':'','BLACK_NG件数':'','構文パス率':'',
    '全体件数':'','正答件数':'','誤答件数':'','正答率':'','BLEU':'','-smooth2':'','-smooth4':'','ROUGE-L':'','Leven':''}

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
    dataset = read_file(file_path)

    #評価用関数実行
    ExactMatch(dataset, results)
    BLEU(dataset, results)
    ROUGE_L(dataset, results)
    Levenstein(dataset, results)

    #結果出力
    try:
        gs_results = []
        for key, value in results.items():
            if value=='':
                gs_results.append('None')
                pass
            else:
                gs_results.append(value)
                print(key,":",value)
        gs.send_gs(gs_results)
    except:
        pass

main()