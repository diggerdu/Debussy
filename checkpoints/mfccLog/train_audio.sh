#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2
expName=mfccLog
selfPath=`realpath $0`
cd "$(git rev-parse --show-toplevel)"
mkdir -p checkpoints/$expName/
cp $selfPath checkpoints/$expName/
python train.py \
 --Path "/home/diggerdu/dataset/tfsrc/train/audio/" \
 --additionPath "/home/diggerdu/dataset/tfsrc/extendTrain/" \
 --dumpPath "data/trainDump" \
 --nClasses 12\
 --name $expName --model pix2pix --which_model_netG wide_resnet_3blocks \
 --nThreads 13\
 --nfft 512 --hop 256 --nFrames 64 --batchSize  4500\
 --split_hop 0 \
 --niter 200 --niter_decay 30 \
 --lr 1e-5 \
 --gpu_ids 0,1,2 \
 --continue_train  --which_epoch 99 \
# --serial_batches
