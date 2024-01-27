from model.transformer import *
from configs.config import *
from utils import *
import logging
from tqdm import tqdm
from random import random

from transformers import T5EncoderModel, Adafactor
from torch.utils.tensorboard import SummaryWriter


class TestTrainer():
    def __init__(self, args):
        # cache some metainformation
        self.config_path = args.config_path
        self.ckpt_path = args.ckpt_path
        self.path2ann = args.path2ann
        self.path2img = args.path2img

        # cache some information
        self.device = args.device
        self.img_size = args.img_size
        self.grad_accumulation_steps = args.grad_accumulation_steps
        self.model_size = args.model_size

        # pytorch mixed precision training data types
        ptdtype = ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
        self.ctx = torch.amp.autocast(device_type=args.device, dtype=ptdtype, enabled=args.amp)
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

        # create the model
        self.model = self.create_model(args.device)

        # load the text encoder
        self.txt_encoder = self.load_text_encoder(args.model_size)

        # configure the optimizer
        self.optimizer = Adafactor(
            params=self.model.parameters(),
            lr=args.max_lr,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )
           
        # configure the learning rate schedule
        self.scheduler = get_lr_scheduler(self.optimizer, args.warmup_iters, args.decay_iters, args.min_lr, args.max_lr)

        # define the loss function
        self.criterion = nn.CrossEntropyLoss()

    def create_model(self, device):
        # Initialize the model
        model = BaseTransformer(BaseTransformerConfig()).to(device)
        return model
    
    def load_text_encoder(self, model_size):
        txt_encoder = T5EncoderModel.from_pretrained(f'google/t5-efficient-{model_size}')
        return txt_encoder

    def train(self, batch_size):

        dataloader = prepare_dataloader(
            model_size=self.model_size,
            img_size=self.img_size,
            config_path=self.config_path,
            ckpt_path=self.ckpt_path,
            batch_size=batch_size,
            path2img=self.path2img,
            path2caps=self.path2ann
        )
        sample = next(iter(dataloader))

        self.model.train() # turn on the model for training

        p = random() # used to stochastically choose which images gets a text embedding

        # transfer the sample tokens to device
        print(f"Shape of image tokens before masking {sample['image_tokens'].shape}")
        img_tokens = sample['image_tokens'].to(self.device)
        mask_img_tokens = mask_tokens(img_tokens)

        # include the text conditionining 90% of the time
        text_emb = self.txt_encoder(sample['text_tokens']).last_hidden_state.to(self.device) if p >= 0.1 else None

        print(f"Shape of img_toknes {img_tokens.shape}")
        if text_emb is not None:

            print(f"Shape of text_emb {text_emb.shape}")

        else:
            print(f"No texts were provided")

        # compute the gradients using mixed precision
        with self.ctx:
            outputs = self.model(mask_img_tokens, text_emb)
            print(f"Output shape before reshaping {outputs.shape}")
            print(f"Target shape before reshaping {img_tokens.shape}")
            print(f"Output shape after reshaping {outputs.view(-1, outputs.size(-1)).shape}")
            print(f"Target shape after reshaping {img_tokens.view(-1).shape}")
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), img_tokens.view(-1))
            loss = loss / self.grad_accumulation_steps

        try:
            # scale the loss
            self.scaler.scale(loss).backward()

            # backward pass process
            self.scaler.step(self.optimizer) # update the optimizer
            self.scaler.update() # update the scale factor
            self.optimizer.zero_grad(set_to_none=True) # zero out the gradients

        except:
            print("Did not successfully pass the backward propagation")

def main(args):
    trainer = TestTrainer(args)
    trainer.train(args.batch_size)

if __name__ == "__main__":
    import argparse
    # setup arguments
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.device = "cpu"

    # information on the vq tokenizer
    args.config_path = "/Users/radiakbar/Projects/muse/configs/vqgan_imagenet_f16_16384/model.yaml"
    args.ckpt_path = "/Users/radiakbar/Projects/muse/configs/vqgan_imagenet_f16_16384/last.ckpt"

    # path of the coco dataset
    args.path2ann = "/Users/radiakbar/Projects/muse/coco_dataset/annotations/captions_val2017.json"
    args.path2img = "/Users/radiakbar/Projects/muse/coco_dataset/val2017"

    # meta info config
    args.img_size = 256
    args.grad_accumulation_steps = 32
    args.seed = 1223

    # training config
    args.dtype = 'bfloat16'
    args.amp = False
    args.model_size = 'base'
    args.min_lr = 1e-5
    args.max_lr = 1e-4
    args.weight_decay = 0.045
    args.warmup_iters = int(5e3)
    args.decay_iters = int(1.5e5)
    args.beta1 = 0.9
    # training run configuration
    args.epochs = 2
    args.batch_size = 16
    args.run_name = "Muse_Implementation"

    main(args)
