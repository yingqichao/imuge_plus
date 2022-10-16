# PAMI
python -m torch.distributed.launch --master_port 3111 --nproc_per_node=4 train.py \
          -opt options/train/train_IRN+_x4.yml -mode 0 --launcher pytorch

# JPEG
#CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --master_port 3040 --nproc_per_node=1 train.py -opt options/train/train_KD_JPEG_x4.yml -val 2 --launcher pytorch

# train
#CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --master_port 3040 --nproc_per_node=1 train.py -opt options/train/train_IRN+_x4.yml -val 0 --launcher pytorch

# eval
#CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --master_port 1750 --nproc_per_node=1 train.py -opt options/train/train_IRN+_x4.yml -val 1 --launcher pytorch