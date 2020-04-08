#!/bin/bash
#
# Copyright 2017 Ubiqus   (Author: Vincent Nguyen)
#		 Systran  (Author: Jean Senellart)
# License MIT
#
# This recipe shows how to build an openNMT translation model for Romance Multi way languages
# based on 200 000 parallel sentences for each pair
#
# Based on the tuto from the OpenNMT forum


# TODO test is GPU is present or not
CUDA_VISIBLE_DEVICES=0
decode_cpu=false

# this is usefull to skip some stages during step by step execution
stage=0

# if you want to run without training and use an existing model in the "exp" folder set notrain to true
notrain=false

output_dir="/output"

# At the moment only "stage" option is available anyway
. local/parse_options.sh


# Preprocess the data - decide here the vocabulary size 50000 default value
if [ $stage -le 0 ]; then
  mkdir -p exp
  echo "$0: preprocessing corpus"
  onmt_preprocess --config preprocess.yml
fi

# Train the model !!!! even if OS cuda device ID is 0 you need -gpuid=1
# Decide here the number of epochs, learning rate, which epoch to start decay, decay rate
# if you change number of epochs do not forget to change the model name too
# This example has a smaller topology compared to tuto for faster training (worse results)
if [ $stage -le 1 ]; then
  if [ $notrain = false ]; then
    echo "$0: training starting, will take a while."
    onmt_train --config train.yml
  else
    echo "$0: using an existing model"
    if [ ! -f /output/model-multi-2-500-600"_final.pt" ]; then
      echo "$0: mode file does not exist"
      exit 1
    fi
  fi
fi

# Deploy model for CPU usage
if [ $stage -le 2 ]; then
  cp -f $(ls -lf $output_dir/model-multi-2-500-600_step_*.pt | sort -t_ -k3n | tail -1) /output/model-multi-2-500-600"_final.pt"
  if [ $decode_cpu = true ]; then
    python3 tools/release_model.py --model /output/model-multi-2-500-600"_final.pt" \
    --output /exp/model-multi-2-500-600"_cpu.pt"
  fi
fi

# Translate using gpu
# you can change this by changing the model name from _final to _cpu and remove -gpu
if [ $stage -le 3 ]; then
  [ $decode_cpu = true ] && dec_opt="" || dec_opt="--gpu 0"
  for src in es fr it pt ro ; do
    for tgt in es fr it pt ro ; do
      [ ! $src = $tgt ] && onmt_translate --replace_unk --model /output/model-multi-2-500-600"_final"*".pt" \
       --src /data/test-$src$tgt.$src.tok --output /output/test-$src$tgt.hyp.$tgt.tok $dec_opt
    done
  done
fi

# Evaluate the generic test set with multi-bleu
if [ $stage -le 4 ]; then
  for src in es fr it pt ro ; do
    for tgt in es fr it pt ro ; do
      [ ! $src = $tgt ] && perl local/multi-bleu.perl /data/test-$src$tgt.$tgt.tok \
      < /output/test-$src$tgt.hyp.$tgt.tok > /output/test-$src$tgt"_multibleu".txt
    done
  done
  grep BLEU /output/*multibleu.txt
fi
