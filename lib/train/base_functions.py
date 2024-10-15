from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
# datasets related
from lib.config.mf_vit.config import cfg
from lib.train.data import sampler, opencv_loader, processing, LTRLoader, video_transformers as vt
import lib.train.data.transforms as tfm
from lib.train.dataset.icpr_mmvpr_track3 import ICPR_MMVPR_Track3
from lib.utils.misc import is_main_process


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE


def names2datasets(name: str, modality, settings, image_loader):
    assert name in ['ICPR_MMVPR_Track3_train', 'ICPR_MMVPR_Track3_test']

    if name == 'ICPR_MMVPR_Track3_train':
        dataset = ICPR_MMVPR_Track3(settings.env.icpr_dir, split='train', modality=modality)
    elif name == 'ICPR_MMVPR_Track3_test':
        dataset = ICPR_MMVPR_Track3(settings.env.icpr_dir, split='test', modality=modality)
    return dataset


def build_dataloaders(cfg, settings):
    # Data transform
    transform_train = vt.Compose([vt.Resize((cfg.DATA.SIZE, cfg.DATA.SIZE), interpolation='bilinear'),
                                  vt.RandomCrop(size=(cfg.DATA.CROP_SIZE, cfg.DATA.CROP_SIZE)),
                                  vt.ClipToTensor(),
                                  vt.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])])

    transform_val = vt.Compose([vt.Resize((cfg.DATA.SIZE, cfg.DATA.SIZE), interpolation='bilinear'),
                                vt.CenterCrop(size=(cfg.DATA.CROP_SIZE, cfg.DATA.CROP_SIZE)),
                                vt.ClipToTensor(),
                                vt.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])

    data_processing_train = processing.Processing(mode='sequence', transform=transform_train)

    data_processing_val = processing.Processing(mode='sequence', transform=transform_val)

    # Train sampler and loader
    dataset_train = sampler.TrackingSampler(
        dataset=names2datasets(cfg.DATA.TRAIN.DATASET_NAME, cfg.DATA.MODALITY, settings, opencv_loader),
        num_frames=cfg.DATA.SAMPLE_FRAMES,
        processing=data_processing_train,
        frame_sample_mode=cfg.DATA.TRAIN.SAMPLER_MODE)

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    # Validation samplers and loaders
    dataset_val = sampler.TrackingSampler(
        dataset=names2datasets(cfg.DATA.VAL.DATASET_NAME, cfg.DATA.MODALITY, settings, opencv_loader),
        num_frames=cfg.DATA.SAMPLE_FRAMES,
        processing=data_processing_val,
        frame_sample_mode=cfg.DATA.VAL.SAMPLER_MODE)
    val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                           num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                           epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    return loader_train, loader_val


def get_optimizer_scheduler(net, cfg):
    train_type = getattr(cfg.TRAIN.PROMPT, "TYPE", [])

    if train_type:
        # print("Only training score_branch. Learnable parameters are shown below.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if any([param in n for param in train_type]) and p.requires_grad]}
        ]

        for n, p in net.named_parameters():
            if all([param not in n for param in train_type]):
                p.requires_grad = False
    else:
        param_dicts = net.parameters()
    total_num = sum(p.numel() for n, p in net.named_parameters())
    trainable_num = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad)
    if is_main_process():
        print("Learnable parameters are shown below.")
        for n, p in net.named_parameters():
            if p.requires_grad:
                print(n, p.numel())
    print('Total: ', total_num, 'Trainable: ', trainable_num)

    if cfg.TRAIN.OPTIMIZER == 'SGD':
        optimizer = torch.optim.SGD(param_dicts, lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == 'ADAM':
        optimizer = torch.optim.Adam(param_dicts, lr=cfg.TRAIN.LR)
    elif cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")

    if cfg.TRAIN.SCHEDULER.TYPE == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.TRAIN.EPOCH - cfg.TRAIN.WARM_UP_EPOCH,
                                                               eta_min=cfg.TRAIN.SCHEDULER.MIN_LR)
    elif cfg.TRAIN.SCHEDULER.TYPE == 'warmup_and_cosine':
        scheduler = WarmupAndCosineAnnealingScheduler(optimizer, cfg.TRAIN.SCHEDULER.INITIAL_LR,
                                                      cfg.TRAIN.LR, cfg.TRAIN.SCHEDULER.MIN_LR,
                                                      cfg.TRAIN.EPOCH, cfg.TRAIN.WARM_UP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                         gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, scheduler


class WarmupAndCosineAnnealingScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, initial_lr, lr_after_warmup, min_lr, total_epochs, warmup_epochs, last_epoch=-1):

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))

        base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        base_lr_ratios = [(base_lr / base_lrs[0]) for base_lr in base_lrs]

        self.warmup_epochs = warmup_epochs
        if warmup_epochs > 0:
            self.initial_lr = initial_lr
            self.lr_after_warmup = lr_after_warmup
            self.warmup_epochs = warmup_epochs
            self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                                      lr_lambda=self.linear_lr_lambda)
            self.warmup_scheduler.base_lrs = [base_lr_ratios[i] * initial_lr for i, base_lr in
                                              enumerate(base_lr_ratios)]
            self.warmup_scheduler._step_count = 0
            self.warmup_scheduler.last_epoch = -1

        eta_min_base = min_lr
        eta_mins = [eta_min_base * base_lr_ratio for base_lr_ratio in base_lr_ratios]
        self.annealing_schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=eta_min)
            for eta_min in eta_mins]
        super().__init__(optimizer, last_epoch)

    def linear_lr_lambda(self, epoch):
        return 1 + (self.lr_after_warmup - self.initial_lr) * epoch / (self.warmup_epochs * self.initial_lr)

    def step(self):
        self.last_epoch += 1
        if self.warmup_epochs > 0 and self.last_epoch <= self.warmup_epochs:
            self.warmup_scheduler.step()
        else:
            if self.last_epoch != 0:
                values = []
                for idx, annealing_scheduler in enumerate(self.annealing_schedulers):
                    # annealing_scheduler.last_epoch = self.last_epoch - self.warmup_epochs
                    annealing_scheduler.step()
                    values.append(annealing_scheduler.get_last_lr()[idx])
                for i, data in enumerate(zip(self.optimizer.param_groups, values)):
                    param_group, lr = data
                    param_group['lr'] = lr
                    self.print_lr(self.verbose, i, lr)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
