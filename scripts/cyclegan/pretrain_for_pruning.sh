python3 train.py \
--dataroot ./database/horse2zebra/ \
--model cyclegan \
--lambda_weight 1e-3 \
--ndf 64  \
--ngf 48 \
--name horse2zebra_ngf24_ndf64_norm1e-3 \
--gpu_ids 0