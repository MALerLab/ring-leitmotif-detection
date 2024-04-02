import torch.nn as nn
from x_transformers.x_transformers import ScaledSinusoidalEmbedding, Encoder
from .baselines import CNNModel

class BBoxModel(CNNModel):
    """
    Input: Waveform (batch, 330750)
    Output: (batch, num_classes, 2)
    """
    def __init__(self,
                 num_classes=21,
                 duration_samples = 646,
                 apply_attn=True,
                 attn_dim=64,
                 attn_depth=3,
                 attn_heads=6):
        super().__init__(num_classes=num_classes)
        self.apply_attn = apply_attn
        if self.apply_attn:
            self.pos_enc = ScaledSinusoidalEmbedding(attn_dim)
            self.encoder = Encoder(dim=attn_dim,
                                depth=attn_depth,
                                heads=attn_heads,
                                attn_dropout=0.2,
                                ff_dropout=0.2)
        self.proj = nn.Linear(attn_dim*duration_samples, num_classes*2)
    
    def forward(self, x):
        cqt = self.transform(x)
        cqt = (cqt / cqt.max()).transpose(1, 2)
        out = self.stack(cqt)
        if self.apply_attn:
            out = out + self.pos_enc(out)
            out = self.encoder(out)
        out = out.flatten(1, 2)
        bbox_pred = self.proj(out)
        bbox_pred = bbox_pred.reshape(bbox_pred.shape[0], -1, 2)
        return bbox_pred