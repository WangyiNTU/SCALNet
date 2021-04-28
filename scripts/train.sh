#!/bin/bash

#Train Simple Counting and Localization Network (SCALNet)
# actual epochs = epochs* 9 (9 iteration for each sample, see sampler.py)
# --preload generate heatmap and density map npz files for fast training. If no use, online generate the maps.
# Refer to train_options for more arguments.
python train.py --model DLANet --dataset NWPU --batch_size 32 --loss LocLoss --gpus 0 --lr 0.0001 --epochs 20 --save_model_interval 2 --preload --save
