CUDA_VISIBLE_DEVICES=3 python pro-mp_run_walker.py
CUDA_VISIBLE_DEVICES=0 python pro-mp_run_cheetah.py
CUDA_VISIBLE_DEVICES=3 python maml_run_cheetah.py

python viskit/frontend.py  ./data/pro-mp/walker --port 5002