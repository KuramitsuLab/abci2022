#!/bin/bash
#$ -l rt_AG.small=1
#$ -l h_rt=03:10:00
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
echo "$me"

START="megagonlabs/t5-base-japanese-web"

python3 rebuild_vocab.py --tokenizer_path="$START" vocab/special_var.txt vocab/pysem5k.vocab

python3 finetune.py\
    --model_path='$START'\
    --tokenizer_path="local"\
    --max_epochs=5\
    --max_time='00:03:00:00'\
    --batch_size=16\
    --auto_batch_size\
    --output_path="pretrained_$me"\
    /groups/gcc50582/possy/code_pretrain/pymsp3_train.jsonl /groups/gcc50582/possy/code_pretrain/kaggle_msp3_valid.jsonl

