from model.transformer import *
from configs.config import *
from utils import *
import logging
from tqdm import tqdm
from random import random

from transformers import T5EncoderModel, Adafactor

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.tensorboard import SummaryWriter


class Trainer():
    def __init__(self, args):

        # cache some information
        self.local_rank = args.local_rank
        self.img_size = args.img_size
        self.vq_size = args.vq_size
        self.grad_accumulation_steps = args.grad_accumulation_steps
        self.model_size = args.model_size

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
        dataloader, l = prepare_dataloader(self.model_size, self.img_size, self.vq_size, batch_size)

        # save the model if it is on the main GPU
        if self.local_rank == 0:
            logger = SummaryWriter(os.path.join('runs', run_name))

        self.model.train() # turn on the model for training

        # Training loop
        for epoch in range(epochs):

            logging.info(f"Starting epoch {epoch + 1} on GPU {self.local_rank}:")
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
                if ((batch_idx + 1) % self.grad_accumulation_steps == 0) or ((batch_idx + 1) == l):
                    self.scaler.step(self.optimizer) # update the optimizer
                    self.scaler.update() # update the scale factor
                    self.optimizer.zero_grad(set_to_none=True) # zero out the gradients

                # update the progress bar   
                pbar.set_postfix(Loss=loss.item())
                logger.add_scalar("Loss", loss.item(), global_step=epoch * l + batch_idx)

                # save the model/checkpoint every 5000 steps for the main GPU
                if (self.local_rank == 0) and (batch_idx + 1) % 10000 == 0:
                    torch.save(self.model.module.state_dict(), os.path.join("models", run_name, f"chkpt.pt"))
        
        # save the model after all epochs are done
        if (self.local_rank == 0):
            torch.save(self.model.module.state_dict(), os.path.join("models", run_name, f"chkpt.pt"))

def main(args):
    init_process_group(args.backends)
    trainer = Trainer(args)
    trainer.train(args.epochs, args.batch_size, args.run_name)
    destroy_process_group()

if __name__ == "__main__":
    import argparse
    # setup arguments
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # distributed training configuration
    args.backends = 'nccl'
    args.local_rank = int(os.environ['LOCAL_RANK']) # local GPU id
    args.world_size = int(os.environ['WORLD_SIZE']) # the number of GPUs in total
    args.device = f'cuda:{args.local_rank}'

    torch.cuda.set_device(args.device)

    # meta info config
    args.img_size = 16
    args.vq_size = 16
    args.grad_accumulation_steps = 32
    args.seed = 1223

    # training config
    args.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    args.amp = True
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

    # run the training
    main(args)