import torch
import random
import os
import math

import yaml
from omegaconf import OmegaConf

# importing vqgan stuff
from taming.models.vqgan import VQModel

# libraries for data-processing
from transformers import T5TokenizerFast
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.datasets.coco import CocoCaptions
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

# libraries for training configuration
from torch.optim.lr_scheduler import LambdaLR

# the class to process tokens
class TokenProcessor:
    def __init__(
        self, 
        encoder_size: str, 
        img_size: int,
        config_path: str,
        ckpt_path: str, 
        device='cpu'
    ) -> None:
        """
        encoder_size: The size of the text encoder (e.g. 'base', 'large', 'small', etc)
        img_size: The size of the square image (base resolution is 256, but super resolution is 512)
        config_path: The path to the VQ Tokenizer
        ckpt_path: The checkpoint weights of the VQ tokenizer
        device: The device used do all the data work (default cpu)
        """

        # cache image size
        self.img_size = img_size

        # load the tokenizers for both image and text
        self.txt_tokenizer = T5TokenizerFast.from_pretrained(encoder_size)

        # load vq model based on config and size
        config_model = self.load_config(config_path)
        with torch.no_grad():
            self.img_tokenizer = self.load_vqgan(config_model, ckpt_path).to(device)

    def load_vqgan(self, config, ckpt_path=None):
        model = VQModel(**config.model.params)
        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            missing, unexpected = model.load_state_dict(sd, strict=False)
        return model.eval()

    def load_config(self, config_path, display=False):
        config = OmegaConf.load(config_path)
        if display:
            print(yaml.dump(OmegaConf.to_container(config)))
        return config

    def to_tokens(self, examples):
        img_batches = []
        txt_batches = []
        for img, caps in examples:
            # random integer to randomly select one of the 5 captions
            idx = random.randint(0, 4)

            z, _, [_, _, indices] = self.img_tokenizer.encode(self.preprocess_vqgan(self.preprocess(img)))

            # append the image tokens and one of the 5 cpations
            img_batches.append(indices.transpose(1, 0))
            txt_batches.append(self.txt_tokenizer(caps[idx], truncation=True, return_tensors='pt').input_ids[0])

        return {"image_tokens": torch.cat(img_batches, dim=0).long(), "text_tokens": pad_sequence(txt_batches, batch_first=True, padding_value=1)}
    
    def preprocess_vqgan(self, x):
        x = 2.*x - 1.
        return x
    
    def preprocess(self, img, target_image_size=256):
        s = min(img.size)
        
        if s < target_image_size:
            s = target_image_size
            
        r = target_image_size / s
        s = (round(r * img.size[1]), round(r * img.size[0]))
        img = TF.resize(img, s, interpolation=Image.LANCZOS)
        img = TF.center_crop(img, output_size=2 * [target_image_size])
        img = torch.unsqueeze(T.ToTensor()(img), 0)
        return img

    # try to transform from pil image into actual image
    def decode_transform(self, x):
        x = x.detach().cpu()
        x = torch.clamp(x, -1., 1.)
        x = (x + 1.)/2.
        x = x.permute(1,2,0).numpy()
        x = (255*x).astype(np.uint8)
        x = Image.fromarray(x)
        if not x.mode == "RGB":
            x = x.convert("RGB")
        return x
    
def prepare_dataloader(
        model_size:str, 
        img_size:int, 
        config_path: str,
        ckpt_path: str, 
        batch_size: int,
        path2img:str,
        path2caps:str
    ) -> None:

    """
    model_size: Size of the text encoder
    img_size: The intended height and width of the image
    vq_size: The divisor for the total number of tokens after tokenized
    batch_size: Batch size of the dataset
    path2img: The path to the COCO images
    path2caps: The path to the captions of the images
    """
    # coco dataset
    ds = CocoCaptions(root=path2img, annFile=path2caps)
    # create the tokenizer and the batcher function
    processor = TokenProcessor(
        encoder_size=f't5-{model_size}',  
        img_size=img_size, 
        config_path=config_path, 
        ckpt_path=ckpt_path
        )
    # create the batched dataset
    dataloader = DataLoader(
        dataset=ds,
        collate_fn=processor.to_tokens,
        batch_size=batch_size
    )

    return dataloader

def CosineAnneallingWarmupLR(iter:int, warmup_iters: int, decay_iters:int, max_lr: float, min_lr: float) -> float:
    """
    iter: Index argument for the learning rate scheduler
    warmup_iters: Number of warmup iterations
    decay_iters: Number of decay iterations
    max_lr: Maximum learning rate at peaks
    min_lr: Minimum leanring rate at troughs
    """

    # 1) linear warmup for warmup_iters steps
    if iter < warmup_iters:
        return max_lr * iter / warmup_iters
    
    # 2) if it > lr_decay_iters, return min learning rate
    if iter > decay_iters:
        return min_lr
    
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iter - warmup_iters) / (decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (max_lr - min_lr)

def mask_tokens(tokens, mask_idx: int=16384):
    """
    tokens: Input tokens
    token_idx: Index of the mask token
    """
    p = 0
    # consistently sample the distribution until a probability that makes sense
    while p >= 1:
        r = random.random()
        p = 2/math.pi * (1 - r**2) ** (-1/2)

    # randomly mask the tokens based on the percentage
    mask = torch.randn(tokens.size()) < p
    tokens[mask] = mask_idx
    return tokens


def get_lr_scheduler(optimizer, warmup_iters, decay_iters, min_lr, max_lr):
    """
    optimizer: Optimizer of the model
    warmup_iters: Warmup iterations
    decay_iters: Decay iterations
    min_lr: Minimum learning rate at troughs
    max_lr: Maximum learning rate at peaks
    """
    # create the linear warmup and cosine decay
    lr_lambda = lambda iter: CosineAnneallingWarmupLR(iter, warmup_iters, decay_iters, max_lr, min_lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler
    
if __name__ == "__main__":
    # test the functionality of the module
    path2data = "/Users/radiakbar/Projects/muse/coco_dataset/val2017"
    path2ann = "/Users/radiakbar/Projects/muse/coco_dataset/annotations/captions_val2017.json"

    loader = prepare_dataloader(
        model_size='base', 
        img_size=256, 
        config_path="configs/vqgan_imagenet_f16_16384/model.yaml",
        ckpt_path="configs/vqgan_imagenet_f16_16384/last.ckpt", 
        batch_size=2,
        path2img=path2data,
        path2caps=path2ann
    )
    print('Length of the dataset: ', len(loader))
    img, cap = next(iter(loader))
    print(img)
    print(mask_tokens(img))
