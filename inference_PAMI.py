import time

def inference_script_PAMI(*, opt, args, rank, model, train_loader, val_loader, train_sampler):
    ####################################################################################################
    # todo: Eval
    # todo: the evaluation procedure should ONLY include evaluate so far
    ####################################################################################################
    if rank <= 0:
        print('Start evaluating... ')
        # if 'COCO' in opt['eval_dataset']:
        #     root = '/groupshare/real_world_test_images'
        # elif 'ILSVRC' in opt['eval_dataset']:
        #     root = '/groupshare/real_world_test_images_ILSVRC'
        # else:
        #     root = '/groupshare/real_world_test_images_CelebA'

        ### test on hand-crafted 3000 images.
        # model.evaluate(
        #     data_origin = os.path.join(root,opt['eval_kind'],'ori_COCO_0114'),
        #     data_immunize = os.path.join(root,opt['eval_kind'],'immu_COCO_0114'),
        #     data_tampered = os.path.join(root,opt['eval_kind'],'tamper_COCO_0114'),
        #     data_tampersource = os.path.join(root,opt['eval_kind'],'tamper_COCO_0114'),
        #     data_mask = os.path.join(root,opt['eval_kind'],'binary_masks_COCO_0114')
        # )
        ### test on minor modification
        model.evaluate(
            data_origin=opt['path']['data_origin'],
            data_immunize=opt['path']['data_immunize'],
            data_tampered=opt['path']['data_tampered'],
            data_tampersource=opt['path']['data_tampersource'],
            data_mask=opt['path']['data_mask']
        )