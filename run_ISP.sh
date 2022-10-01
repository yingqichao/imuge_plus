CUDA_VISIBLE_DEVICES=3,1,2,0 python -m torch.distributed.launch --master_port 3002 --nproc_per_node=4 train.py \
                                -opt options/train/train_ISP.yml -mode 2 -task_name UNet -loading_from UNet \
                                --launcher pytorch
# note: mode 0 represents main task, 1 represents finetuning