#!/usr/bin/env python3

# Example command:
# ./bin/predict.py \
#       model.path=<path to checkpoint, prepared by make_checkpoint.py> \
#       indir=<path to input data> \
#       outdir=<where to store predicts>

import logging
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
import yaml
from omegaconf import OmegaConf

from inpainting_methods.saicinpainting.training.trainers import load_checkpoint

LOGGER = logging.getLogger(__name__)


def solve(mask=None, image=None, checkpoint_path='./big-lama/models/best.ckpt', train_config_path='./big-lama/config.yaml'):
    """
    mask (B,H,W)不要带通道那一维
    image (B,C,H,W)
    输出是一个tensor, (B,C,H,W)
    checkpoint_path 和 train_config_path是需要的，改一下前面的那个'root'，别的不要动。
    """
    mask = torch.rand(2, 1, 64, 64).cuda()
    image = torch.randn(2, 3, 64, 64).cuda()

    # try:
    #     register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        # device = torch.device('cuda')

    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    # out_ext = predict_config.get('out_ext', '.png')

    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    # if not refine == False:
    # if not predict_config.get('refine', False):
    model = model.cuda()

    # if not predict_config.indir.endswith('/'):
    #     predict_config.indir += '/'

    # mask = torch.randn(1500,1200)
    # mask = mask.numpy()
    # print(type(mask))
    # image = torch.randn(3,1500,1200)
    # image = image.numpy()

    # rec = torch.load('/root/lama/tensor_dict.pt') #这个前面的root要改！
    # mask = rec['mask']
    # image = rec['image']
    ans = torch.empty_like(image)

    print(checkpoint_path)
    # for iter in range(image.shape[0]):
    #
    #     dataset = make_default_val_dataset(mask[iter].numpy(), image[iter].numpy(), kind='default',
    #                                        img_suffix='.png', pad_out_to_modulo=8)
    #     for img_i in tqdm.trange(len(dataset)):
    #         # mask_fname = dataset.mask_filenames[img_i]
    #         # cur_out_fname = os.path.join(
    #         #     predict_config.outdir,
    #         #     os.path.splitext(mask_fname[len(predict_config.indir):])[0] + out_ext
    #         # )
    #         # os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
    #         batch = default_collate([dataset[img_i]])
            # print(batch.keys())
            # print(batch['image'])
            # print(batch['mask'])
            # print(batch['unpad_to_size'])
    with torch.no_grad():
        batch = {
            'image': image.cuda(),
            'mask': mask.cuda()
        }
        # batch['mask'] = (batch['mask'] > 0) * 1
        batch = model(batch)

        ans = batch['inpainted']
        # cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
        # unpad_to_size = batch.get('unpad_to_size', None)
        # if unpad_to_size is not None:
        #     orig_height, orig_width = unpad_to_size
        #     cur_res = cur_res[:orig_height, :orig_width]

    # cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    # cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
    # ans[iter] = torch.from_numpy(cur_res).permute(2, 0, 1) / 255.0

    return ans
    # return torch.from_numpy(cur_res)
    # cv2.imwrite(cur_out_fname, cur_res)

    # except KeyboardInterrupt:
    #     LOGGER.warning('Interrupted by user')
    # except Exception as ex:
    #     LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
    #     sys.exit(1)

if __name__ == '__main__':
    # original
    # mask = torch.rand(2,64,64)
    # image = torch.randn(2,3,64,64)
    # now
    inpainted = solve()
    print(inpainted.shape)