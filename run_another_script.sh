CUDA_VISIBLE_DEVICES=1,3,4 python -m torch.distributed.launch --master_port 27766 --nproc_per_node=1 train.py -opt options/train/train_IRNclr_x4.yml -val 0 --launcher pytorch
