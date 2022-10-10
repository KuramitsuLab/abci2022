module load gcc/9.3.0 python/3.8 cuda/11.2 cudnn/8.1
pip3 install -r eval/requirements.txt
python3 eval/eval_py.py abci2022/testfile/tested.jsonl  result
