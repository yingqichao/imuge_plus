python -m torch.distributed.launch --master_port 5499 --nproc_per_node=2 train.py \
                                -opt options/train/ISP/train_ISP_CASIA.yml -mode 9 -task_name RAW_protection_CASIA \
                                --launcher pytorch
