#!/bin/bash
#$ -l rt_G.small=1
#$ -l h_rt=01:30:00
#$-j y
#$-m b
#$-m a
#$-m e
#$-cwd

source /etc/profile.d/modules.sh
module load gcc/9.3.0 python/3.8 cuda/11.2 cudnn/8.1

pip3 install --user --upgrade pip
pip3 install -r requirements.txt

export LD_LIBRARY_PATH=/lib:/usr/lib:/usr/local/lib:/apps/centos7/python/3.8.7/lib

me=`basename "$0" | sed 's/.sh/\n/g' | head -1`

python3 finetune.py\
    --model_path='google/mt5-small'\
    --tokenizer_path='google/mt5-small'\
    --batch_size=16\
    --output_path="model_$me"\
    --tested_file="tweet/$me_tested.jsonl"\
    tweet/NTCIR-13_MedWeb_test.jsonl tweet/NTCIR-13_MedWeb_train.jsonl tweet/NTCIR-13_MedWeb_valid.jsonl

cp "tweet/$me_tested.jsonl" "model_$me/tested.jsonl"

python3 make_xai.py\
    --model_path="model_$me"\
    --batch_size=16\
    --tested_file="tweet/$me_xai_tested.jsonl"\
    tweet/tweet/NTCIR-13_MedWeb_test_xai.jsonl