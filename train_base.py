from model.transformer import *
from configs.config import *
from utils import *
import logging
from tqdm import tqdm
from random import random

from transformers import T5EncoderModel

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter


class Trainer(nn.Module):
    def __init__(self, args):

        # cache some information
        self.local_rank = args.local_rank
        self.img_size = args.img_size
        self.vq_size = args.vq_size
        self.grad_accumulation_steps = args.grad_accumulation_steps

        # setting up the backends calculation
        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # pytorch mixed precision training data types
        ptdtype = ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
        self.ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype, enabled=args.amp)
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

        # create the model
        self.model = self.create_distributed_model(args.local_rank)

        # load the text encoder
        self.txt_encoder = self.load_text_encoder(args.model_size)

        # configure the optimizer
        self.optimizer = configure_optimizer(self.model, args.weight_decay, args.lr, args.beats, args.eps, args.device_type)

        # configure the learning rate schedule
        self.scheduler = get_lr_scheduler(self.optimizer, args.warmup_iters, args.decay_iters, args.min_lr, args.max_lr)

        # define the loss function
        self.criterion = nn.CrossEntropyLoss()

    def create_distributed_model(self, local_rank):
        # Initialize the model
        model = BaseTransformer(BaseTransformerConfig()).to(local_rank)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        return model
    
    def load_text_encoder(self, model_size):
        txt_encoder = T5EncoderModel.from_pretrained(f'google/t5-efficient-{model_size}')
        return txt_encoder

    def train(self, epochs, batch_size, run_name):

        # prepare the dataloader
        dataloader = prepare_dataloader(self.img_size, self.vq_size, batch_size, is_distributed=True)

        # save the model if it is on the main GPU
        if self.local_rank == 0:
            logger = SummaryWriter(os.path.join('runs', run_name))

        l = len(dataloader)
        self.model.train() # turn on the model for training

        # Training loop
        for epoch in range(epochs):

            logging.info(f"Starting epoch {epoch + 1} on GPU {self.local_rank}:")
            dataloader.sampler.set_epoch(epoch) # shuffler for the distributed dataloader
            print(f"Epoch {epoch + 1} on GPU {self.local_rank}: ")
            pbar = tqdm(dataloader)

            for batch_idx, sample in enumerate(dataloader):
                p = random()

                # only include text conditioning 90% of the time
                if p < 0.1:
                    img_tokens, text_emb = sample['image_tokens'].to(self.local_rank), None
                
                else:
                    text_emb = self.txt_encoder(sample['text_tokens']).last_hidden_state.to(self.local_rank)
                    img_tokens = sample['image_tokens'].to(self.local_rank)

                with self.ctx:
                    outputs = self.model(img_tokens, text_emb)
                    loss = self.criterion(outputs.view(-1, outputs.size(-1)), img_tokens.view(-1))
                    loss = loss / self.grad_accumulation_steps

                # scale the loss
                self.scaler.scale(loss).backward()

                # gradient accumulation process
                if ((batch_idx + 1) % self.grad_accumulation_steps == 0) or ((i + 1) == l):
                    self.scaler.step(self.optimizer) # update the optimizer for gradient clipping
                    self.scaler.update() # update the scale factor
                    self.optimizer.zero_grad(set_to_none=True) # zero out the gradients

                # update the progress bar   
                pbar.set_postfix(Loss=loss.item())
                logger.add_scalar("Loss", loss.item(), global_step=epoch * l + batch_idx)

        if (self.local_rank == 0):
            torch.save(self.model.module.state_dict(), os.path.join("models", run_name, f"chkpt.pt"))

        # Save the trained model
        torch.save(self.model.state_dict(), 'path')

if __name__ == "__main__":
    import argparse
    # setup arguments
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # model config arguments
    args.