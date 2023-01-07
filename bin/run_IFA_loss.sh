python -m torch.distributed.launch --master_port 5444 --nproc_per_node=2 train.py \
                                -opt options/train/IFA/train_restormer_restoration.yml -mode 2 -task_name PSNR_predict \
                                --launcher pytorch
