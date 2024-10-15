import csv
import importlib
import os
import argparse
import zipfile
import re

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.nn import functional as F

# datasets related
from lib.models.mf_vit import build_mf_vit
from lib.train.base_functions import names2datasets, update_settings
from lib.train.data import sampler, opencv_loader, processing, LTRLoader, video_transformers as vt
from lib.train.run_training import init_seeds
import lib.train.admin.settings as ws_settings


def main(script_name, config_name, save_dir, local_rank, seed):
    '''Set seed for different process'''
    init_seeds(seed)

    settings = ws_settings.Settings()
    settings.script_name = script_name
    settings.config_name = config_name
    settings.save_dir = os.path.abspath(save_dir)
    settings.local_rank = local_rank
    total_epoch  = re.search(r'ep(\d+)$', config_name)
    if total_epoch:
        total_epoch = int(total_epoch.group(1))
    else:
        raise RuntimeError("Epoch Error: config_name must end with 'epXXX', where XXX are digits.")
    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    settings.cfg_file = os.path.join(prj_dir, 'experiments/%s/%s.yaml' % (script_name, config_name))
    checkpoint_path = os.path.join(prj_dir, save_dir, 'checkpoints/train/mf_vit', config_name,
                                   'VisionTransformer_ep{:04}.pth.tar'.format(total_epoch))

    settings.description = 'Training script for STARK-S, STARK-ST stage1, and STARK-ST stage2'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    transform_test = vt.Compose([vt.Resize((cfg.DATA.SIZE, cfg.DATA.SIZE), interpolation='bilinear'),
                                 vt.CenterCrop(size=(cfg.DATA.CROP_SIZE, cfg.DATA.CROP_SIZE)),
                                 vt.ClipToTensor(),
                                 vt.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])
    data_processing_test = processing.Processing(mode='sequence', transform=transform_test)
    # Validation samplers and loaders
    dataset_val = sampler.TrackingSampler(
        dataset=names2datasets('ICPR_MMVPR_Track3_test', cfg.DATA.MODALITY, settings, opencv_loader),
        num_frames=cfg.DATA.SAMPLE_FRAMES,
        processing=data_processing_test,
        frame_sample_mode=cfg.DATA.VAL.SAMPLER_MODE)
    val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=1,
                           num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                           epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)
    if settings.script_name == "mf_vit":
        net = build_mf_vit(cfg, training=False)
    # torch.load(checkpoint_path, map_location='cpu')['net']
    net.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['net'], strict=True)
    net.cuda()
    net.eval()

    output = []
    with torch.no_grad():
        for i, data in enumerate(loader_val, 1):
            data = data.to('cuda:0')

            all_images = {}
            for key in data.keys():
                if 'images' not in key or (type(data[key]) is list and None in data[key]):
                    continue
                all_images[key.split('_')[0]] = data[key][0].view(-1, *data[key].shape[2:])
            if cfg.DATA.MODALITY == 'RGB':
                out_dict = net(x=all_images['rgb'])
            elif cfg.DATA.MODALITY == 'Depth':
                out_dict = net(x=all_images['depth'])
            elif cfg.DATA.MODALITY == 'IR':
                out_dict = net(x=all_images['ir'])
            elif cfg.DATA.MODALITY == 'RGBT':
                out_dict = net(x=all_images['rgb'], x_tir=all_images['ir'])
            elif cfg.DATA.MODALITY == 'RGBD':
                out_dict = net(x=all_images['rgb'], x_depth=all_images['depth'])
            elif cfg.DATA.MODALITY == 'RTD':
                out_dict = net(x=all_images['rgb'], x_tir=all_images['ir'], x_depth=all_images['depth'])
            else:
                Exception("Unknown modality")

            rst = out_dict['logist']
            rst = rst.reshape(1, 1, -1).mean(1)
            rst = F.softmax(rst, dim=1)
            rst = rst.data.cpu().numpy().copy()

            output.append([i, rst])

    video_pred = [np.argsort(x[1])[0][-5:][::-1] for x in output]
    video_labels = [x[0] for x in output]

    # save ./script_name/config_name/
    output_dir = os.path.join('./submission', script_name, config_name)
    os.makedirs(output_dir, exist_ok=True)

    submission_file = os.path.join(output_dir, 'submission.csv')

    with open(submission_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Video', 'Prediction'])
        for vid_name, pred in zip(video_labels, video_pred):
            pred_str = ' '.join(map(str, pred))
            csvwriter.writerow([vid_name, pred_str])

    # zip
    zip_file = os.path.join(output_dir, 'submission.zip')
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(submission_file, os.path.basename(submission_file))

    print(f'Submission file compressed and saved to {zip_file}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video Prediction using MF_ViT model')
    parser.add_argument('--script_name', type=str, default='mf_vit', help='Script name')
    parser.add_argument('--config_name', type=str, default='track3_rtd_k400_ep10',
                        help='Config name')
    parser.add_argument('--save_dir', type=str, default='./output', help='Directory to save outputs')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    main(args.script_name, args.config_name, args.save_dir, args.local_rank, args.seed)
