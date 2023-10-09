import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.box_utils import make_anchors

class Detection_loss(nn.Module):
    def __init__(self, crop_size=[64, 128, 128], topk=7, spacing=[2.0, 1.0, 1.0]):
        super(Detection_loss, self).__init__()
        self.crop_size = crop_size
        self.topk = topk
        self.spacing = np.array(spacing)

    @staticmethod  
    def cls_loss(pred, target, mask_ignore, alpha = 0.75 , gamma = 2.0, num_neg = 10000, num_hard = 100, ratio = 100):
        classification_losses = []
        batch_size = pred.shape[0]
        for j in range(batch_size):
            pred_b = pred[j]
            target_b = target[j]
            mask_ignore_b = mask_ignore[j]
            cls_prob = torch.sigmoid(pred_b.detach())
            cls_prob = torch.clamp(cls_prob, 1e-4, 1.0 - 1e-4)
            alpha_factor = torch.ones(pred_b.shape).to(pred_b.device) * alpha
            alpha_factor = torch.where(torch.eq(target_b, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(target_b, 1.), 1. - cls_prob, cls_prob)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = F.binary_cross_entropy_with_logits(pred_b, target_b, reduction='none')
            num_positive_pixels = torch.sum(target_b == 1)
            cls_loss = focal_weight * bce
            cls_loss = torch.where(torch.eq(mask_ignore_b, 0), cls_loss, 0)
            record_targets = target_b.clone()
            
            if num_positive_pixels > 0:
                FN_weights = 4.0  # 10.0  for ablation study
                FN_index = torch.lt(cls_prob, 0.8) & (record_targets == 1)  # 0.9
                cls_loss[FN_index == 1] = FN_weights * cls_loss[FN_index == 1]
                Negative_loss = cls_loss[record_targets == 0]
                Positive_loss = cls_loss[record_targets == 1]
                neg_idcs = random.sample(range(len(Negative_loss)), min(num_neg, len(Negative_loss))) 
                Negative_loss = Negative_loss[neg_idcs] 
                _, keep_idx = torch.topk(Negative_loss, ratio * num_positive_pixels) 
                Negative_loss = Negative_loss[keep_idx] 
                Positive_loss = Positive_loss.sum()
                Negative_loss = Negative_loss.sum()
                cls_loss = Positive_loss + Negative_loss

            else:
                Negative_loss = cls_loss[record_targets == 0]
                neg_idcs = random.sample(range(len(Negative_loss)), min(num_neg, len(Negative_loss)))
                Negative_loss = Negative_loss[neg_idcs]
                assert len(Negative_loss) > num_hard
                _, keep_idx = torch.topk(Negative_loss, num_hard)
                Negative_loss = Negative_loss[keep_idx]
                Negative_loss = Negative_loss.sum()
                cls_loss = Negative_loss
            classification_losses.append(cls_loss / torch.clamp(num_positive_pixels.float(), min=1.0))
        return torch.mean(torch.stack(classification_losses))
    
    @staticmethod
    def target_proprocess(annotations, device, input_size, mask_ignore):
        batch_size = len(annotations)
        max_sample_cnt = 1

        for anno in annotations:
            max_sample_cnt = max(max_sample_cnt, len(anno['box']))

        annotations_new = -1 * torch.ones((batch_size, max_sample_cnt, 7)).to(device)
        for j in range(batch_size):
            bbox_annotation_boxes = annotations[j]['box']
            bbox_annotation_target = []
            # z_ctr, y_ctr, x_ctr, d, h, w
            # crop_box = torch.tensor([0., 0., 0., input_size[0], input_size[1], input_size[2]]).to(device)
            for s in range(len(bbox_annotation_boxes)):
                # coordinate z_ctr, y_ctr, x_ctr, d, h, w
                each_label = bbox_annotation_boxes[s]
                # coordinate convert zmin, ymin, xmin, d, h, w
                z1, y1, x1, z2, y2, x2 = each_label
                
                nd = torch.clamp(z2 - z1, min=0.0)
                nh = torch.clamp(y2 - y1, min=0.0)
                nw = torch.clamp(x2 - x1, min=0.0)
                if nd * nh * nw == 0:
                    continue
                # percent = nw * nh * nd / (each_label[3] * each_label[4] * each_label[5])
                # if (percent > 0.1) and (nw*nh*nd >= 15):
                bbox = torch.from_numpy(np.array([float(z1+0.5*nd), float(y1+0.5*nh), float(x1+0.5 * nw), 
                float(nd), float(nh), float(nw), 0])).to(device)
                bbox_annotation_target.append(bbox.view(1, 7))
                # else:
                #     mask_ignore[j, 0, int(z1):int(torch.ceil(z2)), int(y1):int(torch.ceil(y2)), int(x1):int(torch.ceil(x2))] = -1
            if len(bbox_annotation_target) > 0:
                bbox_annotation_target = torch.cat(bbox_annotation_target, 0)
                annotations_new[j, :len(bbox_annotation_target)] = bbox_annotation_target
        # ctr_z, ctr_y, ctr_x, d, h, w, (0 or -1)
        return annotations_new, mask_ignore
    
    @staticmethod
    def bbox_iou(box1, box2, DIoU=True, eps = 1e-7):
        def zyxdhw2zyxzyx(box, dim=-1):
            ctr_zyx, dhw = torch.split(box, 3, dim)
            z1y1x1 = ctr_zyx - dhw/2
            z2y2x2 = ctr_zyx + dhw/2
            return torch.cat((z1y1x1, z2y2x2), dim)  # zyxzyx bbox
        box1 = zyxdhw2zyxzyx(box1)
        box2 = zyxdhw2zyxzyx(box2)
        # Get the coordinates of bounding boxes
        b1_z1, b1_y1, b1_x1, b1_z2, b1_y2, b1_x2 = box1.chunk(6, -1)
        b2_z1, b2_y1, b2_x1, b2_z2, b2_y2, b2_x2 = box2.chunk(6, -1)
        w1, h1, d1 = b1_x2 - b1_x1, b1_y2 - b1_y1, b1_z2 - b1_z1
        w2, h2, d2 = b2_x2 - b2_x1, b2_y2 - b2_y1, b2_z2 - b2_z1

        # Intersection area
        inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
                (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0) * \
                (b1_z2.minimum(b2_z2) - b1_z1.maximum(b2_z1)).clamp(0) + eps

        # Union Area
        union = w1 * h1 * d1 + w2 * h2 * d2 - inter

        # IoU
        iou = inter / union
        if DIoU:
            cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
            ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
            cd = b1_z2.maximum(b2_z2) - b1_z1.minimum(b2_z1)  # convex depth
            c2 = cw ** 2 + ch ** 2 + cd ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2 + 
            + (b2_z1 + b2_z2 - b1_z1 - b1_z2) ** 2) / 4  # center dist ** 2 
            return iou - rho2 / c2  # DIoU
        return iou  # IoU
    
    @staticmethod
    def bbox_decode(anchor_points, pred_offsets, pred_shapes, stride_tensor, dim=-1):
        c_zyx = (anchor_points + pred_offsets) * stride_tensor
        return torch.cat((c_zyx, 2*pred_shapes), dim)  # zyxdhw bbox
    
    @staticmethod
    def get_pos_target(annotations, anchor_points, stride, spacing, topk=7, ignore_ratio=5):# larger the ignore_ratio, the more GPU memory is used
        batchsize, num, _ = annotations.size()
        mask_gt = annotations[:, :, -1].clone().gt_(-1)
        ctr_gt_boxes = annotations[:, :, :3] / stride #z0, y0, x0
        shape = annotations[:, :, 3:6] / 2 # half d h w
        sp = torch.from_numpy(spacing).to(ctr_gt_boxes.device).view(1, 1, 1, 3)
        # distance (b, n_max_object, anchors)
        distance = -(((ctr_gt_boxes.unsqueeze(2) - anchor_points.unsqueeze(0)) * sp).pow(2).sum(-1))
        _, topk_inds = torch.topk(distance, (ignore_ratio + 1) * topk, dim=-1, largest=True, sorted=True)
        mask_topk = F.one_hot(topk_inds[:, :, :topk], distance.size()[-1]).sum(-2)
        mask_ignore = -1 * F.one_hot(topk_inds[:, :, topk:], distance.size()[-1]).sum(-2)
        mask_pos = mask_topk * mask_gt.unsqueeze(-1)
        mask_ignore = mask_ignore * mask_gt.unsqueeze(-1)
        gt_idx = mask_pos.argmax(-2)
        batch_ind = torch.arange(end=batchsize, dtype=torch.int64, device=ctr_gt_boxes.device)[..., None]
        gt_idx = gt_idx + batch_ind * num 
        target_ctr = ctr_gt_boxes.view(-1, 3)[gt_idx]
        target_offset = target_ctr - anchor_points
        target_shape = shape.view(-1, 3)[gt_idx]
        target_bboxes = annotations[:, :, :-1].view(-1, 6)[gt_idx]
        target_scores, _ = torch.max(mask_pos, 1)
        mask_ignore, _ = torch.min(mask_ignore, 1)
        del target_ctr, distance, mask_topk
        return target_offset, target_shape, target_bboxes, target_scores.unsqueeze(-1), mask_ignore.unsqueeze(-1)
    
    def forward(self, output, annotations, device):
        Cls = output['Cls']
        Shape = output['Shape']
        Offset = output['Offset']
        batch_size = Cls.size()[0]
        target_mask_ignore = torch.zeros(Cls.size()).to(device)
        # view shape
        pred_scores = Cls.view(batch_size, 1, -1)
        pred_shapes = Shape.view(batch_size, 3, -1)
        pred_offsets = Offset.view(batch_size, 3, -1)
        # (b, num_points, 1|3)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_shapes = pred_shapes.permute(0, 2, 1).contiguous()
        pred_offsets = pred_offsets.permute(0, 2, 1).contiguous()
        # process annotations   
        process_annotations, target_mask_ignore = self.target_proprocess(annotations, device, self.crop_size, target_mask_ignore)
        target_mask_ignore = target_mask_ignore.view(batch_size, 1,  -1)
        target_mask_ignore = target_mask_ignore.permute(0, 2, 1).contiguous()
        # generate center points. Only support single scale feature
        anchor_points, stride_tensor = make_anchors(Cls, self.crop_size, 0) # z, y, x
        # predict bboxes (zyxdhw)
        pred_bboxes = self.bbox_decode(anchor_points, pred_offsets, pred_shapes, stride_tensor)
        # assigned points and targets (target bboxes zyxdhw)
        target_offset, target_shape, target_bboxes, target_scores, mask_ignore = self.get_pos_target(process_annotations, 
                                                anchor_points, stride_tensor[0].view(1, 1, 3), self.spacing, self.topk)
        # merge mask ignore
        mask_ignore = mask_ignore.bool() | target_mask_ignore.bool()
        fg_mask = target_scores.squeeze(-1).bool()
        classification_losses = self.cls_loss(pred_scores, target_scores, mask_ignore.int())
        
        if fg_mask.sum() == 0:
            reg_losses = torch.tensor(0).float().to(device)
            offset_losses = torch.tensor(0).float().to(device)
            iou_losses = torch.tensor(0).float().to(device)
        else:
            reg_losses = torch.abs(pred_shapes[fg_mask] - target_shape[fg_mask]).mean()
            offset_losses = torch.abs(pred_offsets[fg_mask] - target_offset[fg_mask]).mean()
            iou_losses = -(self.bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])).mean()
        
        return classification_losses, reg_losses, offset_losses, iou_losses