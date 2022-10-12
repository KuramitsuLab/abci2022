import datetime
import sys
import json
import mojimoji
from nltk.translate.bleu_score import corpus_bleu
import re
from janome.tokenizer import Tokenizer
from sumeval.metrics.rouge import RougeCalculator
import Levenshtein
import gs

import warnings
warnings.filterwarnings('ignore')

#前処理
def pretreatment(ref, pred):
    try:
        #全角→半角
        new_ref = mojimoji.zen_to_han(ref)
        new_pred = mojimoji.zen_to_han(pred)
        #置換
        new_ref = new_ref.replace(" ","").replace('<nl>','\n').replace('<tab>','    ')
        new_pred = new_pred.replace(" ","").replace('<nl>','\n').replace('<tab>','    ')
        return ((new_ref, new_pred))
    except:
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

    def tokenize_japanese(japanese):
        try:
            t = Tokenizer()
            tokens = t.tokenize(japanese)
            ss = [token.surface for token in tokens]
            return ss
        except:
            return pattern.split(japanese)

    references = []
    predictions = []
    for ref ,pred in dataset:
        references.append([tokenize_japanese(ref)])
        predictions.append(tokenize_japanese(pred))
    result_bleu = corpus_bleu(references,predictions)*100
    results['BLEU'] = round(result_bleu,3)

def ROUGE_L(dataset,results):
    rouge = RougeCalculator()
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
    ROUGE_L(dataset, results)
    Levenstein(dataset, results)
    
    #結果出力
    try:
        gs_result_list = []
        for key, value in results.items():
            if value=='':
                gs_result_list.append('None')
                pass
            else:
                gs_result_list.append(value)
                print(key,":",value)
        gs.send_gs(gs_result_list)
    except:
        pass

result_black_ng  = 0
main()