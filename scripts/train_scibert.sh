#!/bin/bash

set -eou pipefail

# LANGUAGE=$1
MODEL=$1
TRAIN=$2
DEV=$3
PARAM=$4

readonly ROOT=$(dirname $(realpath -s $0))

cd $ROOT

touch $MODEL.plan
python classifier.py train -m $MODEL -s ../data/all.scibert.npz -t $TRAIN -d $DEV
rm $MODEL.plan