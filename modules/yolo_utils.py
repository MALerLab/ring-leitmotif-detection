import torch
import torch.nn as nn


def get_iou(pred, gt):
    """
    1-dimensional Intersection over Union
    Args:
        pred: (..., 2)
        gt: (..., 2)

    Returns:
        (..., 1)
    """
    pred_x, pred_w = pred[..., 0:1], pred[..., 1:2]
    gt_x, gt_w = gt[..., 0:1], gt[..., 1:2]

    pred_x1 = pred_x - (pred_w / 2)
    pred_x2 = pred_x + (pred_w / 2)
    gt_x1 = gt_x - (gt_w / 2)
    gt_x2 = gt_x + (gt_w / 2)

    i_start = torch.maximum(pred_x1, gt_x1)
    i_end = torch.minimum(pred_x2, gt_x2)

    intersection = torch.clamp(i_end - i_start, min=0)
    union = pred_w + gt_w - intersection

    return intersection / (union + 1e-16)

def grid_to_absolute(input: torch.Tensor, S=11, batched = False):
    """
    Converts x position from grid-relative to absolute

    Args:
        input: (num_anchors, S, 1:x) or (batch, num_anchors, S, 1:x)

    Returns:
        (num_anchors, S, 1) or (batch, num_anchors, S, 1)
    """

    if batched:
        assert len(input.shape) == 4 and input.shape[2] == S
        return (input + torch.arange(S).float().reshape(1, 1, S, 1).to(input.device)) / S
    else:
        assert len(input.shape) == 3 and input.shape[1] == S
        return (input + torch.arange(S).float().reshape(1, S, 1).to(input.device)) / S


def nms(
        pred,
        anchors: torch.Tensor,
        iou_threshold=0.6, 
        conf_threshold=0.05
    ):
    """
    Args:
        pred: raw output from model (batch, num_anchors, S, 3 + C)

    Returns:
        2-dimensional list of boxes represented as [p_o, x(relative to sample), w, class_idx]
    """
    pred = pred.clone().detach()

    num_batches = pred.shape[0]
    pred[..., 0:1] = torch.sigmoid(pred[..., 0:1])
    pred[..., 1:2] = torch.sigmoid(pred[..., 1:2])
    results = []
    for batch_idx in range(num_batches):
        batch_pred = pred[batch_idx]
        anchors = anchors.reshape(3, 1, 1)
        batch_pred = torch.cat(
            [
                batch_pred[..., 0:1],
                grid_to_absolute(batch_pred[..., 1:2]), 
                torch.exp(batch_pred[..., 2:3]) * anchors,
                torch.argmax(batch_pred[..., 3:], dim=-1).float().unsqueeze(-1)
            ], 
            dim=-1
        ) # (*, 4[p_o, x, w, class_idx])
        
        thresh_mask = batch_pred[..., 0] > conf_threshold
        batch_pred = batch_pred[thresh_mask]
        boxes = batch_pred.tolist()
        boxes = sorted(boxes, key=lambda x: x[0], reverse=True)

        result = []
        while boxes:
            cur_box = boxes.pop(0)
            boxes = [
                box
                for box in boxes
                if box[3] != cur_box[3]
                or get_iou(
                    torch.tensor(cur_box[1:3]).to(pred.device),
                    torch.tensor(box[1:3]).to(pred.device)
                ).squeeze().item()
                < iou_threshold
            ]
            result.append(cur_box)
        
        results.append(result)

    return results

def get_acc(
        suppressed_pred:list, 
        gt:torch.Tensor, 
        anchors:torch.Tensor, 
        iou_threshold=0.5,
        conf_threshold=0.5
    ):
    t = gt.clone().detach()
    batch_size = t.shape[0]
    t[..., 1:2] = grid_to_absolute(t[..., 1:2], batched=True)
    anchors = anchors.reshape(1, 3, 1, 1)
    t[..., 2:3] = t[..., 2:3] * anchors

    num_total = 0
    num_correct = 0
    for i in range(batch_size):
        if torch.sum(t[i][t[i][..., 0] == 1]) == 0:
            continue
        p = torch.tensor(suppressed_pred[i]).to(t.device)
        for t_box in t[i][t[i][..., 0] == 1]:
            num_total += 1
            for p_box in p:
                if (get_iou(t_box[1:3], p_box[1:3]).item() > iou_threshold and 
                    p_box[0] > conf_threshold and
                    p_box[3] == t_box[3]):
                    num_correct += 1
                    break
    if num_total == 0:
        return -1
    return num_correct / num_total