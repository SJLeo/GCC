python3 train.py \
--dataroot ./database/sr/ \
--model srgan \
--initial_path ./experiments/sr_ngf24/checkpoints/model_130.pth \
--ngf 24 \
--ndf 64 \
--lambda_scale 1e-2 \
--name sr_ngf24_scale1e-2 \
--gpu_ids 0
