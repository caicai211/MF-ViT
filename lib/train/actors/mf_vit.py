import numpy as np
# from timm.utils import reduce_tensor

from . import BaseActor
from lib.utils.misc import NestedTensor, reduce_dict, reduce_tensor
import torch
import torch.distributed as dist

from ..admin.stats import topk_accuracy


# from lib.utils.merge import merge_template_search
# from ...utils.heapmap_utils import generate_heatmap
# from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate

# def reduce_tensor(tensor: torch.Tensor):
#     rt = tensor.clone()
#     dist.all_reduce(rt, op=dist.ReduceOp.SUM)
#     rt /= dist.get_world_size()  # 总进程数
#     return rt

class MF_ViTActor(BaseActor):
    """ Actor for training MF_ViT models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region

        all_images = {}
        for key in data.keys():
            if 'images' not in key or (type(data[key]) is list and None in data[key]):
                continue
            all_images[key.split('_')[0]] = data[key][0].view(-1, *data[key].shape[2:])
        if self.cfg.DATA.MODALITY == 'RGB':
            out_dict = self.net(x=all_images['rgb'])
        elif self.cfg.DATA.MODALITY == 'Depth':
            out_dict = self.net(x=all_images['depth'])
        elif self.cfg.DATA.MODALITY == 'IR':
            out_dict = self.net(x=all_images['ir'])
        elif self.cfg.DATA.MODALITY == 'RGBT':
            out_dict = self.net(x=all_images['rgb'], x_tir=all_images['ir'])
        elif self.cfg.DATA.MODALITY == 'RGBD':
            out_dict = self.net(x=all_images['rgb'], x_depth=all_images['depth'])
        elif self.cfg.DATA.MODALITY == 'RTD':
            out_dict = self.net(x=all_images['rgb'], x_tir=all_images['ir'], x_depth=all_images['depth'])
        else:
            Exception("Unknown modality")
        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_label = gt_dict['label']  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)

        pre_logist = pred_dict['logist']
        prec1, prec5 = topk_accuracy(pre_logist, gt_label, (1, 5))
        cls_loss = self.objective['cls'](pre_logist, gt_label)  # (BN,4) (BN,4)

        # loss = reduce_tensor(loss)
        # acc = reduce_dict({'Acc': acc})['Acc']
        prec1 = reduce_tensor(prec1)
        prec5 = reduce_tensor(prec5)

        loss = self.loss_weight['cls'] * cls_loss
        if return_status:
            # status for log
            status = {"Top1": prec1.item(),
                      "Top5": prec5.item(),
                      "Loss/cls": loss.item()}
            return loss, status
        else:
            return loss
