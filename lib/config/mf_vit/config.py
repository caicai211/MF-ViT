from easydict import EasyDict as edict
import yaml

"""
Add default config for OSTrack.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
# MODEL.BACKBONE
cfg.MODEL.PRETRAINED = 'checkpoint.pth'
cfg.MODEL.ADAPTER = False
cfg.MODEL.TYPE = 'vit_base_patch16_224'
cfg.MODEL.GLOBAL_POOL = 'avg'

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.PROMPT = edict()
cfg.TRAIN.PROMPT.TYPE = []
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 100
cfg.TRAIN.WARM_UP_EPOCH = 0
cfg.TRAIN.BATCH_SIZE = 1
cfg.TRAIN.NUM_WORKER = 1
cfg.TRAIN.OPTIMIZER = "SGD"
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WEIGHT_DECAY = 0.1

cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.SMOOTHING = 0.1
cfg.TRAIN.PRINT_INTERVAL = 1
cfg.TRAIN.VAL_EPOCH_INTERVAL = 1
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.AMP = False

cfg.TRAIN.CLS_LOSS_USES = 'giou'

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.INITIAL_LR = 1e-8
cfg.TRAIN.SCHEDULER.MIN_LR = 1e-6

# DATA
cfg.DATA = edict()
cfg.DATA.MODALITY = 'RGB'
cfg.DATA.SIZE = 256
cfg.DATA.CROP_SIZE = 224

cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.NUM_CLASSES = 40
cfg.DATA.SAMPLE_FRAMES = 32
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.SAMPLER_MODE = "mean"  # sampling methods
cfg.DATA.TRAIN.DATASET_NAME = 'THU-READ_train1'

# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.SAMPLER_MODE = "mean"  # sampling methods
cfg.DATA.VAL.DATASET_NAME = 'THU-READ_train1'


# TEST
# cfg.TEST = edict()
# cfg.TEST.TEMPLATE_FACTOR = 2.0
# cfg.TEST.TEMPLATE_SIZE = 128
# cfg.TEST.SEARCH_FACTOR = 5.0
# cfg.TEST.SEARCH_SIZE = 320
# cfg.TEST.EPOCH = 500


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename, base_cfg=None):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            _update_config(cfg, exp_config)
