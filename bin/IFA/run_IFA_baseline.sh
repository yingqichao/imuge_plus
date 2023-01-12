python -m torch.distributed.launch --master_port 6001 --nproc_per_node=1 train.py \
                                -opt options/train/IFA/train_IFA_baseline.yml -mode 3 -task_name IFA_baseline \
                                --launcher pytorch