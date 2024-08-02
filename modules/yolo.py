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
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden, out_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(x)

class YOLO(nn.Module):
    """
    Forward input:
        (batch, 1, 646, 128)
    
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
            ConvBlock(256, 512, (3, 3), 2, 1)
        )
        self.mlp = MLP(512, 512, self.num_final_channels)

    def forward(self, x):
        x = self.stack(x)
        x = x.permute(0, 2, 3, 1).squeeze(1)
        return self.mlp(x)
    
class YOLOLoss(nn.Module):
    def __init__(self, B=3, C=20, lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self.B = B
        self.C = C
        self.num_final_channels = (3 * self.B) + self.C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        self.x_selector = torch.tensor([0 + 3 * i for i in range(self.B)])
        self.w_selector = torch.tensor([1 + 3 * i for i in range(self.B)])
        self.conf_selector = torch.tensor([2 + 3 * i for i in range(self.B)])
        
    def forward(self, pred, gt):
        """
        pred, gt: (batch, S=11, 3B + C)
        """

        x_pred, x_gt = pred[:, :, self.x_selector], gt[:, :, self.x_selector]
        w_pred, w_gt = pred[:, :, self.w_selector], gt[:, :, self.w_selector]
        conf_pred, conf_gt = pred[:, :, self.conf_selector], gt[:, :, self.conf_selector]
        class_pred, class_gt = pred[:, :, 3 * self.B:], gt[:, :, 3 * self.B:]

        # ij-mask
        ij_mask = conf_gt > 0
        x_pred[conf_gt == 0] = 0
        w_pred[conf_gt == 0] = 0

        



        return loss_boxes + loss_classes