# single GPU training
#python train.py -opt options/train/train_IRN_x4.yml

# distributed training
# CVPR
python -m torch.distributed.launch --master_port 27766 --nproc_per_node=3 train.py -opt options/train/train_IRNclr_x4.yml -mode 0 --launcher pytorch

# ISP
#CUDA_VISIBLE_DEVICES=4,3 python -m torch.distributed.launch --master_port 3111 --nproc_per_node=2 train.py -opt options/train/train_ISP.yml -mode 0 --launcher pytorch
#CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch train.py --master_port 3111 --nproc_per_node=1 --launcher pytorch -opt options/train/train_IRN+_x4.yml -mode 0


# ICASSP_NOWAY
#python train.py -opt options/train/train_IRNcrop_x4.yml

# ICASSP_RHI
#CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --master_port 11149 --nproc_per_node=1 train.py -opt options/train/train_IRNrhi_x4.yml --launcher pytorch

# generate images
#CUDA_VISIBLE_DEVICES=1,3,4 python -m torch.distributed.launch --master_port 25577 --nproc_per_node=3 train.py -opt options/train/train_IRN+_x4.yml -val 0 --launcher pytorch
