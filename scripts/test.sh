#!/bin/bash
# val search best epoch result by MAE
python testall.py --dataset NWPU --model_name DLANet --no-preload --no-wait --gpus 0 --test_batch_size 1 --model_path 'path to models folder' --search_thr

# generate txt for searching best threshold (thr) on val set. If we set 0.4 thr, the generate txt with thr [0.38,0.39,0.4,0.41,0.42]
python testall.py --dataset NWPU --model_name DLANet --no-preload --no-wait --gpus 0 --test_batch_size 1 --model_path saved_models/000018.h5  --det_thr 0.4 --save_txt

# search the best thr on val set. choose it from output info (F-measure, precision, recall). Takes some minutes. We choose 0.39.
# need to change the pred_folder in eval_search_thr.py if use other models.
python eval/eval_search_thr.py

# add --save to generate demo image on val set using thr of 0.39
python testall.py --dataset NWPU --model_name DLANet --no-preload --no-wait --gpus 0 --test_batch_size 1 --model_path saved_models/000018.h5  --det_thr 0.39 --save

# unlabel testing on the test set
python test_nwpu_loc.py --dataset NWPU_unlabel --model_name DLANet --no-preload --no-wait --gpus 0 --test_batch_size 1 --model_path saved_models/000018.h5 --save_txt --det_thr 0.39