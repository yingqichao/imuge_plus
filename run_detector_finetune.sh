## my_own_elastic
python -m torch.distributed.launch --master_port 2960 --nproc_per_node=1 train.py \
                                -opt options/train/ISP/train_ISP_detector_finetune.yml -mode 6 -task_name OSN_finetune_no_protection_no_attack --launcher pytorch