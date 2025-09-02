#!/bin/bash
# custom config
export TOKENIZERS_PARALLELISM=false
DATA="/home/umaimarahman/datasets/"
TRAINER=LaFTer
CFG=vit_b32
dset="$1"
txt_cls=lafter
CUDA_VISIBLE_DEVICES=0 python /home/umaimarahman/postphd/lafter_medical/meduna.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/"${dset}".yaml \
--config-file configs/trainers/text_cls/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/"${dset}" \
--lr 0.0005 \
--txt_cls ${txt_cls} \

