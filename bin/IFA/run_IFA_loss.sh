python -m torch.distributed.launch --master_port 5555 --nproc_per_node=1 train.py \
                                -opt options/train/IFA/train_restormer_restoration.yml -mode 2 -task_name PSNR_predict_resnet \
                                --launcher pytorch
