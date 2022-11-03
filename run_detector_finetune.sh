## my_own_elastic
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --master_port 2999 --nproc_per_node=1 train.py \
                                -opt options/train/ISP/train_ISP_detector_finetune.yml -mode 6 -task_name resfcn -loading_from resfcn \
                                --launcher pytorch