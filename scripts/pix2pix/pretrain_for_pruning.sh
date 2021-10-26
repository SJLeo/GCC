#!/usr/bin/env bash
python3 train.py \
--dataroot ./database/cityscapes/ \
--model pix2pix \
--lambda_scale 1e-2 \
--ngf 32 \
--ndf 128 \
--name cityscapes_ngf32_ndf128_scale1e-2 \
--gpu_ids 0