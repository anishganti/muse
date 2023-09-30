import torch
from dataclasses import dataclass

@dataclass
class TestBaseTransformerConfig():
    img_size: int = 16
    codebook_size: int = 8193

    hidden_size: int = 512
    text_hidden_size: int = 512
    num_head: int = 8
    num_layer: int = 6
    dropout: float = 0.1

    device = torch.device('cpu')

@dataclass
class TestSuperResTransformerConfig():
    hi_res_img_size: int = 64
    block_size: int = 8
    codebook_size: int = 8193

    hidden_size: int = 512
    text_hidden_size: int = 512
    num_head: int = 8
    num_enc_layer: int = 4
    num_dec_layer: int = 6
    dropout: float = 0.1

    device = torch.device("cpu")