#!/bin/sh
python run_pretraining.py --model_name GBert-pretraining-qGTN1 --num_train_epochs 5 --do_train --graph
python run_gbert.py --model_name GBert-predict-qGTN1 --use_pretrain --pretrain_dir ../saved/GBert-pretraining-qGTN1 --num_train_epochs 5 --do_train --graph

for i in {1..6}
do
python run_pretraining.py --model_name GBert-pretraining-qGTN1 --use_pretrain --pretrain_dir ../saved/GBert-predict-qGTN1 --num_train_epochs 5 --do_train --graph
python run_gbert.py --model_name GBert-predict-qGTN1 --use_pretrain --pretrain_dir ../saved/GBert-pretraining-qGTN1 --num_train_epochs 5 --do_train --graph
done
