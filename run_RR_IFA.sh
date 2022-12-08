python -m torch.distributed.launch --master_port 5999 --nproc_per_node=2 train.py \
                                -opt options/train/IFA/train_RR_IFA.yml -mode 0 -task_name RR_IFA \
                                --launcher pytorch
