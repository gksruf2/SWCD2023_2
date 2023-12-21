#!/bin/bash

#SBATCH --job-name gksruf
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --partition batch_ugrad
#SBATCH -o test_logs2/slurm-%A-%x.out
#SBATCH --time 1-0

#python data/generate_fashion_datasets.py

#python train.py --name=DPTN_fashion --model=DPTN --dataset_mode=fashion --dataroot=./dataset/fashion --batchSize 16 --gpu_id=0
#python train.py --name=DPTN_fashion --model=DPTN --dataset_mode=fashion --dataroot=./dataset/fashion --batchSize 16 --gpu_id=0 --continue_train 
# train 158 / 200 epoch

#python test.py --name=DPTN_fashion --model=DPTN --dataset_mode=fashion --dataroot=./dataset/fashion --which_epoch 100 --results_dir ./results/DPTN_fashion/100 --batchSize 1 --gpu_id=0
#python test.py --name=DPTN_fashion --model=DPTN --dataset_mode=fashion --dataroot=./dataset/fashion --which_epoch 150 --results_dir ./results/DPTN_fashion/150 --batchSize 1 --gpu_id=0
#python test.py --name=DPTN_fashion --model=DPTN --dataset_mode=fashion --dataroot=./dataset/fashion --which_epoch latest --results_dir ./results/DPTN_fashion/158 --batchSize 1 --gpu_id=0

#python -m  metrics.metrics --gt_path=./dataset/fashion/test --distorated_path=./results/DPTN_fashion/100 --fid_real_path=./dataset/fashion/train --name=./fashion
#python -m  metrics.metrics --gt_path=./dataset/fashion/test --distorated_path=./results/DPTN_fashion/150 --fid_real_path=./dataset/fashion/train --name=./fashion
#python -m  metrics.metrics --gt_path=./dataset/fashion/test --distorated_path=./results/DPTN_fashion/158 --fid_real_path=./dataset/fashion/train --name=./fashion

#python test.py --name=DPTN_fashion --model=DPTN --dataset_mode=fashion --dataroot=./dataset/fashion --which_epoch pre_trained --results_dir ./results/DPTN_fashion/pre_trained --batchSize 1 --gpu_id=0
#python -m  metrics.metrics --gt_path=./dataset/fashion/test --distorated_path=./results/DPTN_fashion/pre_trained --fid_real_path=./dataset/fashion/train --name=./fashion

#python test.py --name=DPTN_fashion --model=DPTN --dataset_mode=fashion --dataroot=./dataset/fashion --which_epoch 30 --results_dir ./results/DPTN_fashion/30 --batchSize 1 --gpu_id=0

#python train.py --name=DPTN_fashion --model=DPTN --dataset_mode=fashion --dataroot=./dataset/fashion --batchSize 16 --gpu_id=0 --lr=0.0001 --continue_train 
#Linear Scaling Rule에 따라 lr 0.0002 -> 0.0001
#python test.py --name=DPTN_fashion --model=DPTN --dataset_mode=fashion --dataroot=./dataset/fashion --which_epoch 200 --results_dir ./results/DPTN_fashion/200 --batchSize 1 --gpu_id=0
#python -m  metrics.metrics --gt_path=./dataset/fashion/test --distorated_path=./results/DPTN_fashion/200 --fid_real_path=./dataset/fashion/train --name=./fashion --minibatch=1 --len_images=10000

#python train.py --name=DPTN_fashion --model=DPTN --dataset_mode=fashion --dataroot=./dataset/fashion --batchSize 16 --gpu_id=0 --lr=0.0001
#python test.py --name=DPTN_fashion --model=DPTN --dataset_mode=fashion --dataroot=./dataset/fashion --which_epoch pre_trained --results_dir ./results/DPTN_fashion/pre_trained_train_image --batchSize 1 --gpu_id=0 --phase train

#python train.py --name=DPTN_fashion --model=DPTN --dataset_mode=fashion --dataroot=./dataset/fashion --batchSize 16 --gpu_id=0 --lr=0.0001 --continue_train 
#python test.py --name=DPTN_fashion --model=DPTN --dataset_mode=fashion --dataroot=./dataset/fashion --which_epoch latest --results_dir ./results/DPTN_fashion/smallnet_non_distillation --batchSize 1 --gpu_id=0
#python -m  metrics.metrics --gt_path=./dataset/fashion/test --distorated_path=./results/DPTN_fashion/smallnet_non_distillation --fid_real_path=./dataset/fashion/train --name=./fashion --minibatch=1 --len_images=20000
#python -m  metrics.metrics --gt_path=./dataset/fashion/test --distorated_path=./results/DPTN_fashion/pre_trained --fid_real_path=./dataset/fashion/train --name=./fashion --minibatch=1 --len_images=20000

#python test.py --name=DPTN_fashion --model=DPTN --dataset_mode=fashion --dataroot=./dataset/fashion --which_epoch latest --results_dir ./results/DPTN_fashion/smallnet_train_image --batchSize 1 --gpu_id=0 --phase train
#python train.py --name=DPTN_fashion --model=DPTN --dataset_mode=fashion_distillation --dataroot=./dataset/fashion --batchSize 16 --gpu_id=0 --lr=0.0001 --continue_train 
#python train.py --name=DPTN_fashion --model=DPTN --dataset_mode=fashion_distillation --dataroot=./dataset/fashion --batchSize 16 --gpu_id=0 --lr=0.0001 --continue_train 

#python train.py --name=DPTN_fashion --model=DPTN --dataset_mode=fashion --dataroot=./dataset/fashion --batchSize 16 --gpu_id=0 --lr=0.0001
python test.py --name=DPTN_fashion --model=DPTN --dataset_mode=fashion --dataroot=./dataset/fashion --which_epoch 190 --results_dir ./results/DPTN_fashion/distill_190 --batchSize 1 --gpu_id=0
python test.py --name=DPTN_fashion --model=DPTN --dataset_mode=fashion --dataroot=./dataset/fashion --which_epoch 200 --results_dir ./results/DPTN_fashion/distill_200 --batchSize 1 --gpu_id=0

python -m  metrics.metrics --gt_path=./dataset/fashion/test --distorated_path=./results/DPTN_fashion/distill_190 --fid_real_path=./dataset/fashion/train --name=./fashion --minibatch=1 --len_images=10000
python -m  metrics.metrics --gt_path=./dataset/fashion/test --distorated_path=./results/DPTN_fashion/distill_200 --fid_real_path=./dataset/fashion/train --name=./fashion --minibatch=1 --len_images=10000

exit 0