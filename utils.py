import torch
import random
import os
import math  

# libraries for data-processing
from datasets import load_dataset
from transformers import T5TokenizerFast
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import numpy as np
import requests
import io
from open_muse.muse import VQGANModel, PaellaVQModel
from PIL import Image

# libraries for training configuration
from torch.optim.lr_scheduler import LambdaLR

# the class to process tokens
class TokenProcessor:
    def __init__(self, encoder_size: str, vq_size: int, img_size: int):

        # the image normalizer/recropper to 256 x 256 pixels
        self.encode_transform = transforms.Compose(
            [
                transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(img_size),
                transforms.ToTensor()
            ]
        )

        # cache image size
        self.img_size = img_size

        # load the tokenizers for both image and text
        self.txt_tokenizer = T5TokenizerFast.from_pretrained(encoder_size)

        # load different vq models depending on image size
        if vq_size == 8:
            self.img_tokenizer = PaellaVQModel.from_pretrained('openMUSE/paellavq-f8-8192-laion')

        if vq_size == 16:
            self.img_tokenizer = VQGANModel.from_pretrained('openMUSE/vqgan-f16-8192-laion')

        # randomly insert colors if can't retrieve image
        self.random_colors = ['pink', 'black', 'white', 'blue', 'green', 'yellow']

    def to_tokens(self, examples):
        img_batches = []
        txt_batches = []
        for example in examples:
            # try to get a random color just in case it can't get the link for the image
            idx = random.randint(0, len(self.random_colors) - 1)
            color = self.random_colors[idx]
            working_link = True

            # try to get the image from the link
            try:
                r = requests.get(example['url'], stream=True)
                image = self.encode_transform(Image.open(io.BytesIO(r.content))).unsqueeze(0)
                img_tokens = self.img_tokenizer.encode(image)[1] # some pictures for some reason breaks so if it doesn't work then move to solid

            except:
                working_link = False

            if working_link:
                print("Actual picture")
                img_batches.append(img_tokens)
                txt_batches.append(self.txt_tokenizer(example['caption'], truncation=True, return_tensors='pt').input_ids[0])
            
            # if it can't get the picture, just insert some solid color background
            else:
                print("Solid color")
                idx = random.randint(0, len(self.random_colors) - 1)
                color = self.random_colors[idx]
                image = self.encode_transform(Image.new('RGB', (self.img_size, self.img_size), color=color)).unsqueeze(0)
                caption = f"Solid {color} picture"
                img_batches.append(self.img_tokenizer.encode(image)[1])
                txt_batches.append(self.txt_tokenizer(caption, return_tensors='pt').input_ids[0])

        return {'image_tokens': torch.cat(img_batches, dim=0).long(), 'text_tokens': pad_sequence(txt_batches, batch_first=True, padding_value=1)}

    # try to transform from pil image into actual image
    def decode_transform(self, rec_img):
        rec_img = 2.0 * rec_img - 1.0
        rec_img = torch.clamp(rec_img, -1.0, 1.0)
        rec_img = (rec_img + 1.0) / 2.0
        rec_img *= 255.0
        rec_img = rec_img.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        pil_images = [Image.fromarray(img) for img in rec_img]

        return pil_images
    
def prepare_dataloader(model_size: str, img_size:int, vq_size:int, batch_size: int):
    # dataset 
    ds = load_dataset("laion/laion400m", split="train").to_iterable_dataset().with_format('torch')
    processor = TokenProcessor(f't5-{model_size}', vq_size, img_size)
    # use metainformation on data to find the length of the dataset
    if 401300000 % batch_size > 0:
        l = 401300000 // batch_size

    elif 401300000 % batch_size == 0:
        l = 401300000 // batch_size + 1

    dataloader = DataLoader(
        dataset=ds,
        collate_fn=processor.to_tokens,
        batch_size=batch_size
    )

    return dataloader, l

def CosineAnneallingWarmupLR(iter:int, warmup_iters: int, decay_iters:int, max_lr: float, min_lr: float):

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

def mask_tokens(tokens, token_idx: int=8192):
    # create the masking rate
    r = random.random()
    p = 2/math.pi * (1 - r**2) ** (-1/2)
    print(p)

    # randomly mask the tokens based on the percentage
    mask = torch.randn(tokens.size()) < p
    tokens[mask] = token_idx
    return tokens


def get_lr_scheduler(optimizer, warmup_iters, decay_iters, min_lr, max_lr):
    # create the linear warmup and cosine decay
    lr_lambda = lambda iter: CosineAnneallingWarmupLR(iter, warmup_iters, decay_iters, max_lr, min_lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler

# setup the logging directories
def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    
if __name__ == "__main__":
    loader, l = prepare_dataloader('base', 256, 16, 2)
    print(l)
    sample = next(iter(loader))
    print(sample['image_tokens'].shape)
    print(mask_tokens(sample['image_tokens']).shape)
