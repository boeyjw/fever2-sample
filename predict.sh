#!/usr/bin/env bash

CUDA_DEVICE=0
default_cuda_device=0
model_path=/users/k21190024/study/fact-checking-repos/fever/baseline/data/models/decomposable_attention.tar.gz

# fever paths
# root_dir=/users/k21190024/study/fact-checking-repos/fever/sheffieldnlp/fever2-sample
# staging=$root_dir/data/predstage
# index_path=$root_dir/data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz
# database_path=$root_dir/data/fever/fever.db

# scifact paths
root_dir=/users/k21190024/study/fact-checking-repos/fever/baseline/dumps/fever/baseline
staging=$root_dir/predstage
index_path=$root_dir/index/feverised-scifact-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz
database_path=$root_dir/feverised-scifact.db

# ln -s $root_dir/data data

echo "start evidence retrieval"

python -m fever.evidence.retrieve \
    --index $index_path \
    --database  $database_path\
    --in-file $1 \
    --out-file $staging/ir.$(basename $1) \
    --max-page 5 \
    --max-sent 5 \
    --parallel True \
    --threads 25

echo "start prediction"
# May have to change database path in library file /scratch/users/k21190024/envs/conda/fever-baseline/lib/python3.6/site-packages/fever/reader/fever_reader.py to db associated with the data
python -m allennlp.run predict \
    $model_path \
    $staging/ir.$(basename $1) \
    --output-file $staging/labels.$(basename $1) \
    --predictor fever \
    --include-package fever.reader \
    --cuda-device ${CUDA_DEVICE:-$default_cuda_device} \
    --silent

echo "prepare submission"
python -m fever.submission.prepare \
    --predicted_labels $staging/labels.$(basename $1) \
    --predicted_evidence $staging/ir.$(basename $1) \
    --out_file $2
