## my_own_elastic
CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch --master_port 3020 --nproc_per_node=2 train.py \
                                -opt options/train/train_ISP_ablation_on_RAW.yml -mode 3 -task_name ablation_on_RAW -loading_from UNet \
                                --launcher pytorch
