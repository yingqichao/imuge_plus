python -m torch.distributed.launch --master_port 3004 --nproc_per_node=1 train.py \
                                -opt options/train/ISP/train_ISP_alone.yml -mode 5 -task_name ISP_alone \
                                --launcher pytorch
