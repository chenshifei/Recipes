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

# At the moment only "stage" option is available anyway
. local/parse_options.sh

# Tokenize and prepare the Corpus
if [ $stage -le 0 ]; then
  echo "$0: tokenizing corpus"
  for f in data/train*.?? ; do tools/tokenize.perl < $f > $f.rawtok ; done
  cat data/train*.rawtok | python3 local/learn_bpe.py -s 32000 > data/esfritptro.bpe32000
  for f in data/*-????.?? ; do \
    tools/tokenize.perl -case_feature -joiner_annotate -nparrallel 4 -bpe_model data/esfritptro.bpe32000 < $f > $f.tok
  done
  for set in train valid test ; do rm data/$set-multi.???.tok ; done
  for src in es fr it pt ro ; do
    for tgt in es fr it pt ro ; do
      [ ! $src = $tgt ] && perl -i.bak -pe "s//__opt_tgt_$tgt\xEF\xBF\xA8N /" data/*-$src$tgt.$src.tok
      for set in train valid test ; do
        [ ! $src = $tgt ] && cat data/$set-$src$tgt.$src.tok >> data/$set-multi.src.tok
        [ ! $src = $tgt ] && cat data/$set-$src$tgt.$tgt.tok >> data/$set-multi.tgt.tok
      done
    done
  done
  paste data/valid-multi.src.tok data/valid-multi.tgt.tok | shuf > data/valid-multi.srctgt.tok
  head -2000 data/valid-multi.srctgt.tok | cut -f1 > data/valid-multi2000.src.tok
  head -2000 data/valid-multi.srctgt.tok | cut -f2 > data/valid-multi2000.tgt.tok
fi

# Preprocess the data - decide here the vocabulary size 50000 default value
if [ $stage -le 1 ]; then
  mkdir -p exp
  echo "$0: preprocessing corpus"
  onmt_preprocess -src_vocab_size 50000 -tgt_vocab_size 50000 \
  -train_src data/train-multi.src.tok -train_tgt data/train-multi.tgt.tok \
  -valid_src data/valid-multi2000.src.tok -valid_tgt data/valid-multi2000.tgt.tok \
  -save_data exp/model-multi
fi

# Train the model !!!! even if OS cuda device ID is 0 you need -gpuid=1
# Decide here the number of epochs, learning rate, which epoch to start decay, decay rate
# if you change number of epochs do not forget to change the model name too
# This example has a smaller topology compared to tuto for faster training (worse results)
if [ $stage -le 2 ]; then
  if [ $notrain = false ]; then
    echo "$0: training starting, will take a while."
    onmt_train -layers 2 -rnn_size 500 -brnn -word_vec_size 600 \
    -end_epoch 13 -learning_rate 1 -start_decay_at 5 -learning_rate_decay 0.65 \
    -data  exp/model-multi-train.t7 -save_model exp/model-multi-2-500-600 -gpuid 1
    cp -f exp/model-multi-2-500-600"_epoch13_"*".t7" exp/model-multi-2-500-600"_final.t7"
  else
    echo "$0: using an existing model"
    if [ ! -f exp/model-multi-2-500-600"_final.t7" ]; then
      echo "$0: mode file does not exist"
      exit 1
    fi
  fi
fi

# Deploy model for CPU usage
if [ $stage -le 3 ]; then
  if [ $decode_cpu = true ]; then
    python3 tools/release_model.py -model exp/model-multi-2-500-600"_final.t7" \
    -output exp/model-multi-2-500-600"_cpu.t7"
  fi
fi

# Translate using gpu
# you can change this by changing the model name from _final to _cpu and remove -gpuid 1
if [ $stage -le 4 ]; then
  [ $decode_cpu = true ] && dec_opt="" || dec_opt="-gpuid 1"
  for src in es fr it pt ro ; do
    for tgt in es fr it pt ro ; do
      [ ! $src = $tgt ] && onmt_translate -replace_unk -model exp/model-multi-2-500-600"_final"*".t7" \
       -src data/test-$src$tgt.$src.tok -output exp/test-$src$tgt.hyp.$tgt.tok $dec_opt
    done
  done
fi

# Evaluate the generic test set with multi-bleu
if [ $stage -le 5 ]; then
  for src in es fr it pt ro ; do
    for tgt in es fr it pt ro ; do
      [ ! $src = $tgt ] && local/multi-bleu.perl data/test-$src$tgt.$tgt.tok \
      < exp/test-$src$tgt.hyp.$tgt.tok > exp/test-$src$tgt"_multibleu".txt
    done
  done
  grep BLEU exp/*multibleu.txt
fi
