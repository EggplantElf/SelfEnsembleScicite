#!/bin/bash

set -eou pipefail

# LANGUAGE=$1
MODEL=$1
IN=$2
OUT=$3

readonly ROOT=$(dirname $(realpath -s $0))


cd $ROOT

python classifier.py pred -m $MODEL -s ../data/all.scibert.npz -i $IN -o $OUT
