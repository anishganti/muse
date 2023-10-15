from dataclasses import dataclass
import torch

@dataclass
class BaseTransformerConfig():
    # image specific configuration
    img_size: int = 16
    codebook_size: int = 8193

    # text specific configuration
    text_hidden_size: int = 768 # dimension of t5-xxl

    # model configuration
    hidden_size: int = 768
    num_head: int = 12
    num_layer: int = 18
    dropout: float = 0.1

@dataclass
class BaseSuperResConfig():
    # image specific configuration
    hi_res_img_size: int = 64
    lo_res_img_size: int = 16
    codebook_size: int = 8193

    # text specific configuration
    text_hidden_size: int = 768

    # model configuration
    hidden_size: int = 512
    num_head: int = 8
    num_enc_layer: int = 8
    num_dec_layer: int = 16
    dropout: float = 0.1