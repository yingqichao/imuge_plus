## my_own_elastic
python -m torch.distributed.launch --master_port 3030 --nproc_per_node=1 train.py \
                                -opt options/train/ISP/train_ISP_ablation_on_RAW.yml -mode 3 -task_name ablation_OSN_finish \
                                --launcher pytorch
