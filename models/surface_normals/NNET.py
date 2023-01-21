import torch
import torch.nn as nn
from models.surface_normals.Encoder import Encoder
from models.surface_normals.Decoder import Decoder

class NNET(nn.Module):
    def __init__(self):
        super(NNET, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        return self.decoder.parameters()

    def forward(self, img, **kwargs):
        return self.decoder(self.encoder(img), **kwargs)

