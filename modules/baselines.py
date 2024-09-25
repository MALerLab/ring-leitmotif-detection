import torch
import torch.nn as nn
from torch.autograd import Function
from nnAudio.features.cqt import CQT1992v2
from x_transformers.x_transformers import ScaledSinusoidalEmbedding, Encoder

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
    
class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding,
            dilation
        ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding, 
            dilation
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
class ConvStack(nn.Module):
    def __init__(self, base_hidden=16):
        super().__init__()
        self.stack1 = nn.Sequential(
            ConvBlock(1, base_hidden, (3, 3), (1, 1), "same", (1, 1)),
            ConvBlock(base_hidden, base_hidden*4, (3, 3), (1, 1), "same", (1, 1)),
            DilatedMaxPool2d((3, 3), (1, 3), (1, 1)),
            ConvBlock(base_hidden*4, base_hidden*8, (3, 3), (1, 1), "same", (3, 1)),
            ConvBlock(base_hidden*8, base_hidden*4, (3, 3), (1, 1), "same", (3, 1)),
            DilatedMaxPool2d((3, 3), (1, 3), (3, 1)),
            ConvBlock(base_hidden*4, base_hidden*16, (3, 3), (1, 1), "same", (9, 1)),
            ConvBlock(base_hidden*16, base_hidden*8, (3, 3), (1, 1), "same", (9, 1)),
            DilatedMaxPool2d((3, 3), (1, 3), (9, 1)),
            ConvBlock(base_hidden*8, base_hidden*16, (1, 4), (1, 1), 0, (1, 1)),
        )
        self.stack2 = nn.Sequential(
            nn.Conv1d(base_hidden*16, base_hidden*32, 3, 1, "same", 27),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(base_hidden*32),
            nn.Conv1d(base_hidden*32, base_hidden*32, 3, 1, "same", 27),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(base_hidden*32),
            DilatedMaxPool1d(3, 1, 27)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.stack1(x)
        x = x.squeeze(3)
        x = self.stack2(x)
        return x.transpose(1, 2)

class CNNModel(nn.Module):
    def __init__(self, num_classes=21, base_hidden=16, dropout=0.2):
        super().__init__()
        self.transform = CQT1992v2()
        self.stack = ConvStack(base_hidden)
        self.proj = nn.Sequential(
            nn.Linear(base_hidden*32, base_hidden*32),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(base_hidden*32, num_classes)
        )

    def forward(self, x):
        cqt = self.transform(x)
        cqt = (cqt / cqt.max()).transpose(1, 2)
        cnn_out = self.stack(cqt)
        pred = self.proj(cnn_out).sigmoid()
        return pred
    
class CRNNModel(nn.Module):
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
        return leitmotif_pred


class CNNAttnModel(CNNModel):
    def __init__(self,
                 num_classes=21,
                 attn_dim=64,
                 attn_depth=3,
                 attn_heads=6):
        super().__init__(num_classes)
        self.pos_enc = ScaledSinusoidalEmbedding(attn_dim)
        self.encoder = Encoder(dim=attn_dim,
                               depth=attn_depth,
                               heads=attn_heads,
                               attn_dropout=0.2,
                               ff_dropout=0.2)
    
    def forward(self, x):
        cqt = self.transform(x)
        cqt = (cqt / cqt.max()).transpose(1, 2)
        cnn_out = self.stack(cqt)
        cnn_out = cnn_out + self.pos_enc(cnn_out)
        enc_out = self.encoder(cnn_out)
        leitmotif_pred = self.proj(enc_out).sigmoid()
        return leitmotif_pred