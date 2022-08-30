CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch --master_port 11149 --nproc_per_node=2 train.py -opt options/train/train_IRNclr_x4.yml --launcher pytorch
