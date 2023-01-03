python -m torch.distributed.launch --master_port 3021 --nproc_per_node=1 train.py \
                                -opt options/train/train_ISP_OSN_baseline.yml -mode 4 \
                                --launcher pytorch
