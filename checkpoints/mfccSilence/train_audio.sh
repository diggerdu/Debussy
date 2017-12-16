#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,2
expName=mfccSilence
selfPath=`realpath $0`
cd "$(git rev-parse --show-toplevel)"
mkdir -p checkpoints/$expName/
cp $selfPath checkpoints/$expName/
python train.py \
 --Path "/home/diggerdu/dataset/tfsrc/train/audio/" \
 --dumpPath "data/trainDump" \
 --nClasses 12\
 --name $expName --model pix2pix --which_model_netG wide_resnet_3blocks \
 --nThreads 13\
 --nfft 512 --hop 256 --nFrames 64 --batchSize  2200\
 --split_hop 0 \
 --niter 100000000000000000000000000000000000 --niter_decay 30 \
 --lr 1e-4 \
 --gpu_ids 0,1 \
 --continue_train \
 --which_epoch 20 \
#  --serial_batches
