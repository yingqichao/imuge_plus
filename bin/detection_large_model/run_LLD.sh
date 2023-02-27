python -m torch.distributed.launch --master_port 6001 --nproc_per_node=2 train.py \
                                -opt options/train/detection_large_model/train_detection_large_model_base.yml -mode 0 -task_name tianchi_0227 \
                                --launcher pytorch