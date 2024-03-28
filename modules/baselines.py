import torch
import torch.nn as nn
from torch.autograd import Function
from nnAudio.features.cqt import CQT1992v2

class DilatedMaxPool2d(nn.Module):
    """
    Dilated MaxPool2d that keeps the time dimension size.
    """
    def __init__(self, kernel_size, stride, dilation):
        super().__init__()
        self.dilation = dilation
        self.maxpool = nn.MaxPool2d(kernel_size, stride, 0, dilation, ceil_mode=True)
    
    def forward(self, x):
        padded_x = torch.zeros((x.shape[0], x.shape[1], x.shape[2] + 2 * self.dilation[0], x.shape[3]), device=x.device)
        padded_x[:, :, self.dilation[0]:-self.dilation[0], :] = x
        return self.maxpool(padded_x)
    
class DilatedMaxPool1d(nn.Module):
    """
    Dilated MaxPool1d that keeps the dimension size.
    """
    def __init__(self, kernel_size, stride, dilation):
        super().__init__()
        self.dilation = dilation
        self.maxpool = nn.MaxPool1d(kernel_size, stride, 0, dilation, ceil_mode=True)
    
    def forward(self, x):
        padded_x = torch.zeros((x.shape[0], x.shape[1], x.shape[2] + 2 * self.dilation), device=x.device)
        padded_x[:, :, self.dilation:-self.dilation] = x
        return self.maxpool(padded_x)
    
class ConvStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack1 = nn.Sequential(
            nn.Conv2d(1, 128, (3, 3), (1, 1), "same", (1, 1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, (3, 3), (1, 1), "same", (1, 1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            DilatedMaxPool2d((3, 3), (1, 3), (1, 1)),
            nn.Conv2d(64, 128, (3, 3), (1, 1), "same", (3, 1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, (3, 3), (1, 1), "same", (3, 1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            DilatedMaxPool2d((3, 3), (1, 3), (3, 1)),
            nn.Conv2d(64, 128, (3, 3), (1, 1), "same", (9, 1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, (3, 3), (1, 1), "same", (9, 1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            DilatedMaxPool2d((3, 3), (1, 3), (9, 1)),
            nn.Conv2d(64, 64, (1, 4), (1, 1), 0, (1, 1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
        )
        self.stack2 = nn.Sequential(
            nn.Conv1d(64, 128, 3, 1, "same", 27),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 64, 3, 1, "same", 27),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            DilatedMaxPool1d(3, 1, 27)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.stack1(x)
        x = x.squeeze(3)
        x = self.stack2(x)
        return x.transpose(1, 2)

# Gradient reversal layer from: https://github.com/tadeephuy/GradientReversal
class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha*grad_output
        return grad_input, None
revgrad = GradientReversal.apply

class GradientReversal(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)
    
class RNNModel(torch.nn.Module):
    def __init__(self, 
                 input_size=84, 
                 hidden_size=128,
                 num_layers=3,
                 num_classes=21):
        super().__init__()
        self.transform = CQT1992v2()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=num_layers, bidirectional=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        self.proj = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        cqt = self.transform(x)
        cqt = (cqt / cqt.max()).transpose(1, 2)
        lstm_out, _ = self.lstm(cqt)
        lstm_out = lstm_out.transpose(1, 2)
        lstm_out = self.batch_norm(lstm_out)
        lstm_out = lstm_out.transpose(1, 2)
        leitmotif_pred = self.proj(lstm_out).sigmoid()
        return leitmotif_pred, None, None
    

class CNNModel(torch.nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.transform = CQT1992v2()
        self.stack = ConvStack()
        self.proj = nn.Linear(64, num_classes)

    def forward(self, x):
        cqt = self.transform(x)
        cqt = (cqt / cqt.max()).transpose(1, 2)
        cnn_out = self.stack(cqt)
        leitmotif_pred = self.proj(cnn_out).sigmoid()
        return leitmotif_pred, None, None
    
class CRNNModel(torch.nn.Module):
    def __init__(self, 
                 input_size=84, 
                 hidden_size=128,
                 num_layers=3,
                 num_classes=21):
        super().__init__()
        self.transform = CQT1992v2()
        self.stack = ConvStack()
        self.lstm = nn.LSTM(64, hidden_size, batch_first=True, num_layers=num_layers)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.proj = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        cqt = self.transform(x)
        cqt = (cqt / cqt.max()).transpose(1, 2)
        cnn_out = self.stack(cqt)
        lstm_out, _ = self.lstm(cnn_out)
        lstm_out = lstm_out.transpose(1, 2)
        lstm_out = self.batch_norm(lstm_out)
        lstm_out = lstm_out.transpose(1, 2)
        leitmotif_pred = self.proj(lstm_out).sigmoid()
        return leitmotif_pred, None, None

class RNNAdvModel(torch.nn.Module):
    def __init__(self, 
                 input_size=84, 
                 hidden_size=128,
                 mlp_hidden_size='default',
                 num_layers=3, 
                 num_versions=16,
                 adv_grad_multiplier=0.01,
                 num_classes=21):
        super().__init__()
        self.transform = CQT1992v2()
        if mlp_hidden_size == 'default':
            mlp_hidden_size = 64
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=num_layers, bidirectional=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        self.proj = nn.Linear(hidden_size * 2, num_classes)
        self.singing_mlp = nn.Sequential(
            GradientReversal(alpha=adv_grad_multiplier),
            nn.Linear(hidden_size * 2, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, 1),
            nn.Sigmoid()
        )
        self.version_mlp = nn.Sequential(
            GradientReversal(alpha=adv_grad_multiplier),
            nn.Linear(hidden_size * 2, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, num_versions)
        )
    
    def freeze_backbone(self):
        self.lstm.requires_grad_(False)
        self.batch_norm.requires_grad_(False)
        self.proj.requires_grad_(False)

    def unfreeze_backbone(self):
        self.lstm.requires_grad_(True)
        self.batch_norm.requires_grad_(True)
        self.proj.requires_grad_(True)
    
    def forward(self, x):
        cqt = self.transform(x)
        cqt = (cqt / cqt.max()).transpose(1, 2)
        lstm_out, _ = self.lstm(cqt)
        lstm_out = lstm_out.transpose(1, 2)
        lstm_out = self.batch_norm(lstm_out)
        lstm_out = lstm_out.transpose(1, 2)
        leitmotif_pred = self.proj(lstm_out).sigmoid()
        singing_pred = self.singing_mlp(lstm_out)
        version_pred = self.version_mlp(lstm_out)
        return leitmotif_pred, singing_pred, version_pred
    
class CNNAdvModel(torch.nn.Module):
    def __init__(self,
                 num_versions=16,
                 adv_grad_multiplier=0.01,
                 mlp_hidden_size='default',
                 num_classes=21):
        super().__init__()
        self.transform = CQT1992v2()
        if mlp_hidden_size == 'default':
            mlp_hidden_size = 64
        self.stack = ConvStack()
        self.proj = nn.Linear(64, num_classes)
        self.singing_mlp = nn.Sequential(
            GradientReversal(alpha=adv_grad_multiplier),
            nn.Linear(64, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, 1),
            nn.Sigmoid()
        )
        self.version_mlp = nn.Sequential(
            GradientReversal(alpha=adv_grad_multiplier),
            nn.Linear(64, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, num_versions)
        )
    
    def freeze_backbone(self):
        self.convstack.stack1.requires_grad_(False)
        self.convstack.stack2.requires_grad_(False)

    def unfreeze_backbone(self):
        self.convstack.stack1.requires_grad_(True)
        self.convstack.stack2.requires_grad_(True)
    
    def forward(self, x):
        cqt = self.transform(x)
        cqt = (cqt / cqt.max()).transpose(1, 2)
        cnn_out = self.stack(cqt)
        leitmotif_pred = self.proj(cnn_out).sigmoid()
        singing_pred = self.singing_mlp(cnn_out)
        version_pred = self.version_mlp(cnn_out)
        return leitmotif_pred, singing_pred, version_pred