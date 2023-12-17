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

        # cache some information
        self.device = args.device
        self.img_size = args.img_size
        self.vq_size = args.vq_size
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

        dataloader, l = prepare_dataloader(self.model_size, self.img_size, self.vq_size, batch_size)
        sample = next(iter(dataloader)) # get 1 sample from the loader

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

    # training parameters
    parser.add_argument("batch_size", type=int, help="The batch size for the training")
    parser.add_argument("--grad_accumulation_steps", type=int, default=32, help="The number of accumulation step")

    # create a list of arguments for the training setup
    parser.add_argument("--device", type=str, default="mps", help="The device used for the whole training process")
    
    # mixed training arguments
    parser.add_argument("--amp", type=bool, default=False, help="Turn on mixed-precision training")
    parser.add_argument("--dtype", type=str, default="float32", help="The floating point used for mixed-precision training")

    # vq-tokenizer parameters
    parser.add_argument("--img_size", type=int, default=256, help="The image size of the vq tokenizer")
    parser.add_argument("--vq_size", type=int, default=16, help="The divisor for the vq tokenizer")
    parser.add_argument("--model_size", type=str, default='base', help="The size of the T5 embedder")

    # training configuration
    parser.add_argument("--min_lr", type=float, default=1e-5, help="The minimum learning rate for the schedule")
    parser.add_argument("--max_lr", type=float, default=1e-4, help="The maximum learning rate of the schedule")
    parser.add_argument("--weight_decay", type=float, default=0.045, help="The weight decay for the optimizer")
    parser.add_argument("--warmup_iters", type=int, default=int(5e3), help="The number of warmup iteration for the scheduler")
    parser.add_argument("--decay_iters", type=int, default=int(1.5e5), help="The number of decay iterations for the scheduler")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta for the optimizer")

    args = parser.parse_args()

    main(args)
