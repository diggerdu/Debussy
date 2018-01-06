#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2
expName=mfccLog
selfPath=`realpath $0`
cd "$(git rev-parse --show-toplevel)"
mkdir -p checkpoints/$expName/
cp $selfPath checkpoints/$expName/
python test.py \
 --which_epoch 210\
 --serial_batches \
 --Path "/home/diggerdu/dataset/tfsrc/test/audio" \
 --dumpPath "data/testDump" \
 --nClasses 12\
 --name $expName --model pix2pix --which_model_netG cnn \
 --nThreads 13 \
 --nfft 512 --hop 256 --nFrames 64 --batchSize  1000\
 --split_hop 0 \
 --gpu_ids 0,1,2\
