import torch
import torch.nn as nn
from nnAudio.features.cqt import CQT1992v2
from .baselines import ConvStack
from x_transformers.x_transformers import ScaledSinusoidalEmbedding, Encoder

class FiLMGenerator(nn.Module):
    '''
    Generates gamma and beta for given input.\n
    Input: (Batch, 1)\n
    Output: (Batch, 2, res_num_blocks, res_hidden_dim)
    '''

    def __init__(self,
                 num_vocab=21,
                 emb_dim=64,
                 hidden_dim=128,
                 num_layers=1,
                 total_conv_channels=0):
        super().__init__()
        self.num_layers = num_layers
        self.emb = nn.Embedding(num_vocab, emb_dim)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(emb_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        self.proj = nn.Linear(hidden_dim, 2 * total_conv_channels)

    def forward(self, x):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x)
        x = self.proj(x)
        x = x.view(x.size(0), 2, -1)
        return x


class FiLM(nn.Module):
    '''
    Conv input: (Batch, C, H, W)\n
    Gamma/Beta input: (Batch, res_hidden_dim)\n
    '''

    def __init__(self):
        super().__init__()

    def forward(self, x, gamma, beta):
        gamma = gamma.unsqueeze(2).unsqueeze(3).expand_as(x)
        beta = beta.unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * gamma + beta


class FiLM1d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, gamma, beta):
        gamma = gamma.unsqueeze(2).expand_as(x)
        beta = beta.unsqueeze(2).expand_as(x)
        return x * gamma + beta


class ResBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 1, padding=0)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.film = FiLM()
        self.relu = nn.ReLU()

    def forward(self, x, gamma, beta):
        x = self.relu(self.conv1(x))
        residual = x
        x = self.bn(self.conv2(x))
        x = self.relu(self.film(x, gamma, beta))
        x = x + residual
        return x


class FiLMModel(torch.nn.Module):
    def __init__(self,
                 num_classes=21,
                 filmgen_emb=64,
                 filmgen_hidden=64):
        super().__init__()
        self.transform = CQT1992v2()
        self.stack = ConvStack()
        self.proj = nn.Linear(64, 1)
        total_conv_channels = sum([layer.out_channels for layer in self.stack.stack1 if isinstance(
            layer, nn.Conv2d)]) + sum([layer.out_channels for layer in self.stack.stack2 if isinstance(layer, nn.Conv1d)])
        self.film_gen = FiLMGenerator(num_vocab=num_classes,
                                      emb_dim=filmgen_emb,
                                      hidden_dim=filmgen_hidden,
                                      num_layers=2,
                                      total_conv_channels=total_conv_channels)
        self.film = FiLM()
        self.film1d = FiLM1d()

    def cnn_forward(self, cqt, labels):
        film_params = self.film_gen(labels)
        x = cqt.unsqueeze(1)

        idx = 0
        for layer in self.stack.stack1:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                x = self.film(x,
                              gamma=film_params[:, 0, idx:idx+layer.out_channels],
                              beta=film_params[:, 1, idx:idx+layer.out_channels])
                idx += layer.out_channels

        x = x.squeeze(3)
        for layer in self.stack.stack2:
            x = layer(x)
            if isinstance(layer, nn.Conv1d):
                x = self.film1d(x,
                                gamma=film_params[:, 0, idx:idx+layer.out_channels],
                                beta=film_params[:, 1, idx:idx+layer.out_channels])
                idx += layer.out_channels
        cnn_out = x.transpose(1, 2)
        return cnn_out

    def forward(self, x, labels):
        cqt = self.transform(x)
        cqt = (cqt / cqt.max()).transpose(1, 2)
        cnn_out = self.cnn_forward(cqt, labels)
        leitmotif_pred = self.proj(cnn_out).sigmoid()
        return leitmotif_pred


class FiLMAttnModel(FiLMModel):
    def __init__(self,
                 num_classes=21,
                 filmgen_emb=64,
                 filmgen_hidden=64,
                 attn_dim=64,
                 attn_depth=3,
                 attn_heads=6):
        super().__init__(num_classes,
                         filmgen_emb,
                         filmgen_hidden)
        self.pos_enc = ScaledSinusoidalEmbedding(attn_dim)
        self.encoder = Encoder(dim=attn_dim,
                               depth=attn_depth,
                               heads=attn_heads,
                               attn_dropout=0.2,
                               ff_dropout=0.2)
        self.bias_for_emb = nn.Parameter(torch.zeros(attn_dim))

    def forward(self, x, labels):
        cqt = self.transform(x)
        cqt = (cqt / cqt.max()).transpose(1, 2)
        cnn_out = self.cnn_forward(cqt, labels)
        cnn_out = cnn_out + self.pos_enc(cnn_out)

        # Add separate embeddings for class label
        label_emb = self.film_gen.emb(labels)
        label_emb += self.bias_for_emb.unsqueeze(0)
        cat_emb = torch.cat([label_emb.unsqueeze(1), cnn_out], dim=1)

        enc_out = self.encoder(cat_emb)
        leitmotif_pred = self.proj(enc_out).sigmoid()
        return leitmotif_pred[:, 1:]