python -m torch.distributed.launch --master_port 27868 --nproc_per_node=1 train.py \
                                -opt options/train/ISP/train_ISP_my_own_elastic.yml -mode 2 -task_name finished_OSN \
                                --launcher pytorch
