python -m torch.distributed.launch --master_port 3000 --nproc_per_node=1 train.py \
                                -opt options/train/ISP/train_ISP_my_own_elastic.yml -mode 5 -task_name my_own_elastic \
                                --launcher pytorch
