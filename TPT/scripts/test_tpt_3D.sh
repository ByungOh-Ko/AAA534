#!/bin/bash

data_root='/hub_data3/byungoh/modelnet40_ply_hdf5_2048/'
testsets=ModelNet40
ctx_init=a_photo_of_a

for arch in 'ViT-B/16' 'RN50'; do
    for n_ctx in '4' '8' '16' '32'; do
        for selection_p in '0.1' '0.3' '0.5' '0.7' '1.0'; do
            for tta_steps in '1' '2' '4' '8'; do
                EXP_NAME=TPT_3D_${arch}_nctx_${n_ctx}_selectionp_${selection_p}_ttasteps_${tta_steps}.json
                CUDA_VISIBLE_DEVICES=1 python ./tpt_classification_3D.py \
                    ${data_root} \
                    --test_sets ModelNet40 \
                    --gpu 0 \
                    -b 1 \
                    --tpt \
                    -a ${arch} \
                    --n_ctx ${n_ctx} \
                    --ctx_init ${ctx_init} \
                    --selection_p ${selection_p} \
                    -p 20 \
                    --output_filepath /hub_data3/byungoh/TPT/outputs/${EXP_NAME}
            done
        done
    done
done