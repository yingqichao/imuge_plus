
### train
CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.launch --master_port 3002 --nproc_per_node=3 train.py \
                                -opt options/train/train_ISP.yml -mode 2 -task_name UNet -loading_from UNet \
                                --launcher pytorch


### eval
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 3002 --nproc_per_node=1 train.py \
#                                -opt options/train/train_ISP.yml -mode 1 -task_name UNet -loading_from UNet \
#                                --launcher pytorch

# note: mode 0 represents main task, 1 represents finetuning