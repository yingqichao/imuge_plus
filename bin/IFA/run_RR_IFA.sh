python -m torch.distributed.launch --master_port 5699 --nproc_per_node=2 train.py -opt options/train/IFA/train_IFA_seg_post.yml -mode 0 -task_name segment_postprocess --launcher pytorch
