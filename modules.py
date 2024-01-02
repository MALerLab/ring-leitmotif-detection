import torch
import torch.nn as nn

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
    
class RNNModel(torch.nn.Module):
    def __init__(self, input_size=84, hidden_size=128, num_layers=3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, num_layers=num_layers, bidirectional=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        self.proj = nn.Linear(hidden_size * 2, 21)
    
    def forward(self, x):
        x, _ = self.gru(x)
        x = x.transpose(1, 2)
        x = self.batch_norm(x)
        x = x.transpose(1, 2)
        x = self.proj(x)
        return x.sigmoid()
    
class CNNModel(torch.nn.Module):
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
        self.proj = nn.Linear(64, 21)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.stack1(x)
        x = x.squeeze(3)
        x = self.stack2(x)
        x = x.transpose(1, 2)
        x = self.proj(x)
        return x.sigmoid()