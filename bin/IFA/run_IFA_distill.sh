python -m torch.distributed.launch --master_port 3441 --nproc_per_node=2 train.py -opt options/train/IFA/train_IFA_distill.yml -mode 4 -task_name IFA_distill --launcher pytorch
