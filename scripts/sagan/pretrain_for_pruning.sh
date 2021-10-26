#!/usr/bin/env bash
python3 train.py \
--dataroot ./database/celeb/ \
--model sagan \
--lambda_scale 1e-2 \
--ngf 48 \
--ndf 64 \
--name celeb_ngf48_scale1e-2 \
--gpu_ids 0