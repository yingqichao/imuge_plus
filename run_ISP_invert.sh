python -m torch.distributed.launch --master_port 5499 --nproc_per_node=2 train.py \
                                -opt options/train/ISP/train_ISP_invert.yml -mode 8 -task_name invert_RGB_to_RAW \
                                --launcher pytorch
