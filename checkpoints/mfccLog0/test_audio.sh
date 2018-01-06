#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
expName=mfccLog0
selfPath=`realpath $0`
cd "$(git rev-parse --show-toplevel)"
mkdir -p checkpoints/$expName/
cp $selfPath checkpoints/$expName/
python test.py \
 --which_epoch 293\
 --serial_batches \
 --Path "/home/diggerdu/dataset/tfsrc/test/tempo0.75" \
 --dumpPath "data/testDump" \
 --nClasses 12\
 --name $expName --model pix2pix --which_model_netG wide_resnet_3blocks \
 --nThreads 6 \
 --nfft 512 --hop 256 --nFrames 64 --batchSize  1000\
 --split_hop 0 \
 --gpu_ids 0\
