import torch
import torch.nn as nn
from nnAudio.features.cqt import CQT1992v2
from .yolo_utils import get_iou


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
    def __init__(self, in_features, hidden, num_anchors, C, dropout):
        super().__init__()
        self.num_anchors = num_anchors
        self.C = C
        self.fc = nn.Linear(in_features, hidden)
        self.relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, num_anchors * (3 + C))

    def forward(self, x):
        assert len(x.shape) == 4 and x.shape[3] == 1
        x = x.permute(0, 2, 3, 1).squeeze(2)
        x = self.relu(self.fc(x))
        x = self.fc2(self.dropout(x))
        x = x.reshape(x.shape[0], x.shape[1], self.num_anchors, -1)
        return x.permute(0, 2, 1, 3)

class YOLO(nn.Module):
    """
    Forward input:
        (batch, num_audio_samples)
    
    Forward output:
        (batch, S=11, 3B + C)
    """
    def __init__(self, num_anchors=3, C=20, base_hidden=16, dropout=0):
        super().__init__()
        self.num_final_channels = (3 * num_anchors) + C

        self.transform = CQT1992v2()
        self.stack = nn.Sequential(
            ConvBlock(1, base_hidden, (7, 7), 2, 3),
            ConvBlock(base_hidden, base_hidden*2, (3, 3), 2, 1),
            ConvBlock(base_hidden*2, base_hidden*4, (3, 3), 1, 1),
            ConvBlock(base_hidden*4, base_hidden*2, (1, 1), 1, 0), # reduce
            ConvBlock(base_hidden*2, base_hidden*4, (3, 3), 2, 1),
            ConvBlock(base_hidden*4, base_hidden*8, (3, 3), (2, 1), 1),
            ConvBlock(base_hidden*8, base_hidden*4, (1, 1), 1, 0), # reduce
            ConvBlock(base_hidden*4, base_hidden*8, (3, 3), 2, 1),
            ConvBlock(base_hidden*8, base_hidden*16, (3, 3), 2, 1),
            ConvBlock(base_hidden*16, base_hidden*32, (3, 3), 1, (1, 0))
        )
        self.mlp = MLP(base_hidden*32, base_hidden*32, num_anchors, C, dropout)

    def forward(self, x):
        x = self.transform(x)
        x = (x / x.max()).transpose(1, 2).unsqueeze(1)
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
                 anchors: torch.Tensor,
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
        
    def forward(self, p, t):
        pred = p.clone()
        gt = t.clone()
        obj_mask = gt[..., 0] == 1
        noobj_mask = gt[..., 0] == 0

        # NoObject Loss
        loss_noobj = self.bce(self.sigmoid(pred[..., 0:1][noobj_mask]), gt[..., 0:1][noobj_mask])

        if obj_mask.sum() == 0:
            return (
                self.lambda_noobj * loss_noobj,
                {
                    "noobj": loss_noobj,
                    "obj": torch.tensor(0),
                    "coord": torch.tensor(0),
                    "class": torch.tensor(0)
                }
            )

        # Object Loss
        anchors = self.anchors.reshape(1, 3, 1, 1)
        boundaries_pred = torch.cat([self.sigmoid(pred[..., 1:2]), torch.exp(pred[..., 2:3]) * anchors], dim=-1) # (batch, num_anchors, S, 2)
        ious = get_iou(boundaries_pred[obj_mask], gt[..., 1:3][obj_mask]).detach()
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