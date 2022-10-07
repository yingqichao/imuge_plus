
### train normal
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 3001 --nproc_per_node=2 train.py \
                                -opt options/train/train_ISP.yml -mode 2 -task_name UNet -loading_from UNet \
                                --launcher pytorch

## my_own_elastic or UNet
### eval
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 3002 --nproc_per_node=1 train.py \
#                                -opt options/train/train_ISP.yml -mode 1 -task_name UNet -loading_from UNet \
#                                --launcher pytorch

# note: mode 0 represents main task, 1 represents finetuning