python -m torch.distributed.launch --master_port 6001 --nproc_per_node=2 train.py \
                                -opt options/train/tianchi/train_tianchi.yml -mode 0 -task_name tianchi_baseline \
                                --launcher pytorch