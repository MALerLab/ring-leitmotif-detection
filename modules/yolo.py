import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
class MLP(nn.Module):
    """
    Input:
        (batch, in_features, S, 1)

    Output:
        (batch, num_anchors, S, 3+C)
    """
    def __init__(self, in_features, hidden, num_anchors, C):
        super().__init__()
        self.num_anchors = num_anchors
        self.C = C
        self.fc = nn.Linear(in_features, hidden)
        self.relu = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(hidden, num_anchors * (3 + C))

    def forward(self, x):
        assert len(x.shape) == 4 and x.shape[3] == 1
        x = x.permute(0, 2, 3, 1).squeeze(2)
        x = self.fc2(self.relu(self.fc(x)))
        x = x.reshape(x.shape[0], x.shape[1], self.num_anchors, -1)
        return x.permute(0, 2, 1, 3)

class YOLO(nn.Module):
    """
    Forward input:
        (batch, 1, 646, 84)
    
    Forward output:
        (batch, S=11, 3B + C)
    """
    def __init__(self, B=3, C=20):
        super().__init__()
        self.B = B
        self.C = C
        self.num_final_channels = (3 * self.B) + self.C
        self.stack = nn.Sequential(
            ConvBlock(1, 16, (7, 7), 2, 3),
            ConvBlock(16, 32, (3, 3), 2, 1),
            ConvBlock(32, 64, (3, 3), 1, 1),
            ConvBlock(64, 32, (1, 1), 1, 0), # reduce
            ConvBlock(32, 64, (3, 3), 2, 1),
            ConvBlock(64, 128, (3, 3), (2, 1), 1),
            ConvBlock(128, 64, (1, 1), 1, 0), # reduce
            ConvBlock(64, 128, (3, 3), 2, 1),
            ConvBlock(128, 256, (3, 3), 2, 1),
            ConvBlock(256, 512, (3, 3), 1, (1, 0))
        )
        self.mlp = MLP(512, 512, self.num_final_channels)

    def forward(self, x):
        x = self.stack(x)
        return self.mlp(x)
    
class YOLOLoss(nn.Module):
    """
    1-dimensional YOLO-like loss

    Shapes:
        pred: (batch, num_anchors, S=11, 3[p_o, x, w] + C)
        gt:   (batch, num_anchors, S=11, 4[p_o, x, w, class_idx])
    """
    def __init__(self,
                 anchors,
                 lambda_class=1,
                 lambda_noobj=1,
                 lambda_obj=10,
                 lambda_coord=10):
        super().__init__()
        self.lambda_class = lambda_class
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_coord = lambda_coord
        self.anchors = anchors

        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

    def get_iou(self, pred, gt):
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

        pred_x1 = pred_x - pred_w / 2
        pred_x2 = pred_x + pred_w / 2
        gt_x1 = gt_x - gt_w / 2
        gt_x2 = gt_x + gt_w / 2

        x1 = torch.minimum(pred_x1, gt_x1)
        x2 = torch.maximum(pred_x2, gt_x2)

        intersection = torch.clamp(x2 - x1, min=0)
        union = pred_w + gt_w - intersection

        return intersection / (union + 1e-16)
        
    def forward(self, pred, gt):
        obj_mask = gt[..., 0] == 1
        noobj_mask = gt[..., 0] == 0

        # NoObject Loss
        loss_noobj = self.bce(pred[..., 0:1][noobj_mask], gt[..., 0:1][noobj_mask])

        # Object Loss
        anchors = self.anchors.reshape(1, 3, 1, 1)
        boundaries_pred = torch.cat([self.sigmoid(pred[..., 1:2]), torch.exp(pred[..., 2:3]) * anchors], dim=-1) # (batch, num_anchors, S, 2)
        ious = self.get_iou(boundaries_pred[obj_mask], gt[..., 1:3][obj_mask]).detach()
        loss_obj = self.mse(self.sigmoid(pred[..., 0:1][obj_mask]), ious * gt[..., 0:1][obj_mask])

        # Coordinate Loss
        pred[..., 1:2] = self.sigmoid(pred[..., 1:2]) # x
        gt[..., 2:3] = torch.log(1e-16 + gt[..., 2:3] / anchors) # w
        loss_coord = self.mse(pred[..., 1:3][obj_mask], gt[..., 1:3][obj_mask])
 
        # Class Loss
        loss_class = self.ce(pred[..., 3:][obj_mask], gt[..., 3][obj_mask].long())

        return (
            self.lambda_noobj * loss_noobj + 
            self.lambda_obj * loss_obj + 
            self.lambda_coord * loss_coord + 
            self.lambda_class * loss_class,
            {
                "noobj": loss_noobj,
                "obj": loss_obj,
                "coord": loss_coord,
                "class": loss_class
            }
        )

if __name__ == "__main__":
    anchors = torch.tensor([1, 1, 1])
    loss = YOLOLoss(anchors)
    pred = torch.tensor(
        [
            [
                [100, -100, 0, 100,   0, 0],
                [  0, -100, 0,   0, 100, 0],
                [  0, -100, 0, 100,   0, 0],
                [100, -100, 0,   0, 100, 0]
            ],
            [
                [100, -100, 0, 100,   0, 0],
                [  0, -100, 0,   0, 100, 0],
                [  0, -100, 0, 100,   0, 0],
                [100, -100, 0,   0, 100, 0]
            ],
            [
                [100, -100, 0, 100,   0, 0],
                [  0, -100, 0,   0, 100, 0],
                [  0, -100, 0, 100,   0, 0],
                [100, -100, 0,   0, 100, 0]
            ]
        ]
    ).unsqueeze(0).float()
    gt = torch.tensor(
        [
            [
                [1, 0, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 0],
                [1, 0, 1, 1]
            ],
            [
                [1, 0, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 0],
                [1, 0, 1, 1]
            ],
            [
                [1, 0, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 0],
                [1, 0, 1, 1]
            ]
        ]
    ).unsqueeze(0).float()
    print(pred.shape, gt.shape)
    print(loss(pred, gt))