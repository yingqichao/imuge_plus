CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch --master_port 3004 --nproc_per_node=2 train.py \
                                -opt options/train/train_ISP_alone.yml -mode 5 -task_name ISP_alone -loading_from UNet \
                                --launcher pytorch
