#!/bin/bash
work_path=$(dirname $0)
# partialconv_input_ch4.pth  partialconv.pth
python -m torch.distributed.launch --nproc_per_node=1 main.py \
    --config $work_path/config.yaml --launcher pytorch \
     --load-pretrain pretrains/G1000000.pth
