python -m torch.distributed.launch --master_port 6001 --nproc_per_node=2 train.py \
                                -opt options/train/IFA/train_restormer_restoration.yml -mode 0 -task_name RR_IFA \
                                --launcher pytorch
