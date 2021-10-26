#!/usr/bin/env bash
python3 train.py \
--dataroot ./database/cityscapes/ \
--model pix2pix \
--ngf 32 \
--ndf 128 \
--darts_discriminator \
--arch_lr 1e-4 \
--arch_lr_step \
--scale_prune \
--target_budget 3.0 \
--pretrain_path ./experiments/cityscapes_ngf32_ndf128_scale1e-2/checkpoints/model_best_BtoA.pth \
--online_distillation \
--lambda_content 50 \
--lambda_gram 1e4 \
--name cityscapes_ngf32scale3.0_darts_c50g1e4 \
--gpu_ids 0