python -m torch.distributed.launch --master_port 5499 --nproc_per_node=1 train.py \
                                -opt options/train/ISP/train_ISP_my_own_elastic.yml -mode 7 -task_name test \
                                --launcher pytorch
