## my_own_elastic
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 3002 --nproc_per_node=2 train.py \
                                -opt options/train/train_ISP_my_own_elastic.yml -mode 2 -task_name my_own_elastic -loading_from UNet \
                                --launcher pytorch
