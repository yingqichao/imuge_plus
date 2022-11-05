## my_own_elastic
python -m torch.distributed.launch --master_port 2958 --nproc_per_node=1 train.py \
                                -opt options/train/ISP/train_ISP_detector_finetune.yml -mode 6 -task_name Resfcn --launcher pytorch