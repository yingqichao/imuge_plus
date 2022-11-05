python -m torch.distributed.launch --master_port 3021 --nproc_per_node=1 train.py \
                                -opt options/train/ISP/train_ISP_test.yml -mode 4 -task_name my_own_elastic \
                                --launcher pytorch
