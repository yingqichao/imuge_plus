
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --master_port 4090 --nproc_per_node=1 train.py \
                                -opt options/train/train_resfcn.yml -mode 6 -task_name resfcn \
                                --launcher pytorch
