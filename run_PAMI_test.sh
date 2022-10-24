# eval
python -m torch.distributed.launch --master_port 1000 --nproc_per_node=1 train.py -opt options/test/PAMI/test_PAMI.yml -mode 1 --launcher pytorch