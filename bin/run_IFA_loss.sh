python -m torch.distributed.launch --master_port 5999 --nproc_per_node=1 train_and_test_scripts/train.py \
                                -opt options/train/IFA/train_restormer_restoration.yml -mode 1 -task_name Restormer_restoration \
                                --launcher pytorch
