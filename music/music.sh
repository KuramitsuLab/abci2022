#!/bin/bash
#$ -l rt_G.small=1
#$ -l h_rt=02:00:00
#$-j y
#$-m b
#$-m a
#$-m e
#$-cwd

source /etc/profile.d/modules.sh
module load gcc/9.3.0 python/3.8 cuda/11.2 cudnn/8.1

export LD_LIBRARY_PATH=/lib:/usr/lib:/usr/local/lib:/apps/centos7/python/3.8.7/lib

python3 finetune.py\
    --batch_size=16\
    --output_path='testfile'\
    music_test.jsonl music_train.jsonl