import os

import torch
from timm import create_model

from lib.models.mf_vit.mf_vit import load_state_dict


def build_mf_vit(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    model = create_model(
        model_name=cfg.MODEL.TYPE,
        pretrained=False,
        num_classes=cfg.DATA.NUM_CLASSES,
        all_frames=cfg.DATA.SAMPLE_FRAMES,
        tubelet_size=2,
        fc_drop_rate=0.0,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_block_rate=None,
        use_checkpoint=False,
        use_mean_pooling=True,
        init_scale=0.001,
        add_adapter=cfg.MODEL.ADAPTER)
    if cfg.MODEL.PRETRAINED and training:
        ckpt_path = os.path.join(pretrained_path, cfg.MODEL.PRETRAINED)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        print("Load ckpt from %s" % ckpt_path)
        checkpoint_model = None
        for model_key in 'model|module|net'.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        else:
            checkpoint_model = checkpoint
        # interpolate position embedding
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]  # channel dim
            num_patches = model.patch_embed.num_patches  #
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches  # 0/1
            num_frames = model.patch_embed.num_frames
            # height (== width) for the checkpoint position embedding
            orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (
                    num_frames // model.patch_embed.tubelet_size)) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int((num_patches // (num_frames // model.patch_embed.tubelet_size)) ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> BT, H, W, C -> BT, C, H, W
                pos_tokens = pos_tokens.reshape(-1, num_frames // model.patch_embed.tubelet_size, orig_size, orig_size,
                                                embedding_size)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, num_frames // model.patch_embed.tubelet_size,
                                                                    new_size, new_size, embedding_size)
                pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        load_state_dict(model, checkpoint_model, prefix='')
    return model
