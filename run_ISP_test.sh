## eval
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 3000 --nproc_per_node=1 train.py \
                                -opt options/train/train_ISP_test.yml -mode 0 -task_name UNet -loading_from UNet \
                                --launcher pytorch