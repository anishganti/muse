from modules import *
from random import random
from math import pi
from dataclasses import dataclass

"""
Transformer components for the MUSE model
"""

class SuperResTransformer(nn.Module):
    def __init__(self, config) -> None:
        super(SuperResTransformer, self).__init__()

        # cache shape information
        self.src_seq_len = config.src_height * config.src_width
        self.tgt_seq_len = config.tgt_height * config.tgt_width
        self.img_sz = config.tgt_height
        self.mask_id = config.mask_id
        self.block_size = config.block_size
        
        # creating the transformer
        self.transformer = nn.ModuleDict(dict(
            embedding = nn.Embedding(config.codebook_sz, config.emb_dim),
            src_pos_enc2d = nn.Embedding(config.src_height * config.src_width, config.emb_dim),
            tgt_pos_enc2d = nn.Embedding(config.tgt_height * config.tgt_width, config.emb_dim),
            encoder = nn.ModuleList(
                [EncoderBlock(config.emb_dim, config.nhead, config.dropout) for _ in range(config.n_enc_layer)]
            ),
            decoder = nn.ModuleList(
                [MultiAxisTransformerBlock(config.emb_dim, config.block_size, config.nhead, config.dropout) for _ in range(config.n_dec_layer)],
            ),
            ln_f = LayerNorm(config.emb_dim)
        ))
        # other components of the transformer model
        self.dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.emb_dim, config.codebook_sz)
        # the text encoder components from t5
        self.text_encoder = get_text_encoder(config.t5_model)
        self.proj_text = nn.Linear(config.text_emb_dim, config.emb_dim)

    def get_mask(self):
        p = 2/pi * (1 - random() ** 2) ** (-1/2)
        mask = (torch.randn(self.tgt_seq_len) < p)
        self.register_buffer("mask_tokens", mask, persistent=False)
        return mask

    def forward(self, low_res, hi_res, text=None):
        n, m, device = self.src_seq_len, self.tgt_seq_len, low_res.device

        # encode and project the text into our model's dimension
        if text is not None:
            text_emb = self.text_encoder(text)
            text_emb = self.proj_text(text_emb.last_hidden_state)

        # embed the image and inject positional information
        # NEEDS TO BE CHANGED TO USE THE EMBEDDING FROM THE BASE MODEL
        src_emb = self.transformer.embedding(low_res.flatten(start_dim=1))
        src_emb = self.dropout(src_emb + self.transformer.src_pos_enc2d(torch.arange(n, device=device)))
        
        # pass the source embedding to encoders
        for enc in self.transformer.encoder:
            src_emb = enc(src_emb)

        # concat the low-res images with the text embedding only if the text is not none
        if text is not None:
            src_emb = torch.cat((src_emb, text_emb), dim=-2)

        # embed the hi-res tokens
        # NEEDS TO BE CHANGED TO USE THE EMBEDDING FROM THE BASE MODEL
        tgt_emb = self.transformer.embedding(hi_res)
        tgt_emb = self.dropout(tgt_emb + self.transformer.tgt_pos_enc2d(torch.arange(m, device=device).view(1, self.img_sz, self.img_sz)))
        tgt_emb = img_partitioner(tgt_emb, self.block_size, tgt_emb.shape[-1])

        # pass the target embedding to the transformers
        for dec in self.transformer.decoder:
            tgt_emb = dec(tgt_emb, src_emb)
        tgt_emb = self.transformer.ln_f(tgt_emb)

        return self.lm_head(tgt_emb)
    
    def forward_with_cond_scale(self, low_res, hi_res, text, cond_scale:float):
        # entirely conditioned logits
        if cond_scale == 1.:
            return self.forward(low_res, hi_res, text)
        
        logits = self.forward(low_res, hi_res, text) # logits conditioned on text
        null_logits = self.forward(low_res, hi_res, text=None) # logits with no text conditioning

        # cfg logits
        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        return scaled_logits

    def forward_with_neg_prompt(self, low_res, hi_res, pos_text, neg_text, cond_scale:float):
        # logits with negative prompts
        neg_logits = self.forward(low_res, hi_res, neg_text)
        # logits with positive prompts
        pos_logits = self.forward(low_res, hi_res, pos_text)

        scaled_logits = neg_logits + (pos_logits - neg_logits) * cond_scale

        return scaled_logits

class BaseTransformer(nn.Module):
    def __init__(self, config) -> None:
        super(BaseTransformer, self).__init__()
        # save the sequence length of the base images
        self.height = config.height
        self.width = config.width
        self.seq_len = config.height * config.width
        self.mask_id = config.mask_id

        # create the transformer model
        self.transformer = nn.ModuleDict(dict(
            img_embedding = nn.Embedding(config.codebook_sz, config.emb_dim),
            pos_enc2d = nn.Embedding(config.height * config.width, config.emb_dim),
            decoder = nn.ModuleList(
                [TransformerBlock(config.emb_dim, config.nhead, config.dropout) for _ in range(config.n_layer)]
            ),
            ln_f = LayerNorm(config.emb_dim)
        ))
        # other components of the transformer model
        self.dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.emb_dim, config.codebook_sz)
        # the text encoder components from t5
        self.text_encoder = get_text_encoder(config.t5_model)
        self.proj_text = nn.Linear(config.text_emb_dim, config.emb_dim)

    def get_mask(self):
        p = 2/pi * (1 - random() ** 2) ** (-1/2)
        mask = (torch.randn(self.seq_len) < p)
        self.register_buffer("mask_tokens", mask, persistent=False)
        return mask

    def forward(self, img, text):
        n, device = self.seq_len, img.device
        # encode and project the text into our model's dimension
        # only do this 90% of the time
        if text is not None:
            text_emb = self.text_encoder(text)
            text_emb = self.proj_text(text_emb.last_hidden_state)
        else:
            text_emb = text

        # embed the image and inject positional information
        # mask the tokens based on the muse strategy
        # flatten the image tokens for computation
        img_emb = self.transformer.img_embedding(img.flatten(start_dim=1).masked_fill(self.get_mask(), self.mask_id))
        img_emb = self.dropout(img_emb + self.transformer.pos_enc2d(torch.arange(n, device=device)))
        # pass the image embedding through the transformer block
        for block in self.transformer.decoder:
            img_emb = block(img_emb, text_emb)

        img_emb = self.transformer.ln_f(img_emb)

        return self.lm_head(img_emb)
    
    def forward_with_cond_scale(self, img, text, cond_scale:float):
        # entirely conditioned logits
        if cond_scale == 1.:
            return self.forward(img, text)
        
        logits = self.forward(img, text) # logits conditioned on text
        null_logits = self.forward(img, None) # logits with no text conditioning

        # cfg logits
        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        return scaled_logits
    
    def forward_with_neg_prompt(self, img, pos_text, neg_text, cond_scale:float):
        # logits with negative prompts
        neg_logits = self.forward(img, neg_text)
        # logits with positive prompts
        pos_logits = self.forward(img, pos_text)

        scaled_logits = neg_logits + (pos_logits - neg_logits) * cond_scale

        return scaled_logits
    
@dataclass
class BaseTransformerConfig:
    # image information
    height = 16
    width = 16
    codebook_sz = 1024
    emb_dim = 512
    nhead = 8
    dropout = 0.1
    n_layer = 6
    t5_model = 't5-small'
    text_emb_dim = 512
    mask_id = 1

@dataclass
class SuperResTransformerConfig:
    src_height = 16
    src_width = 16
    tgt_height = 64
    tgt_width = 64
    emb_dim = 256
    codebook_sz = 1024
    mask_id = 1
    nhead = 4
    block_size = 8
    n_enc_layer = 8
    n_dec_layer = 8
    dropout = 0.1
    t5_model = 't5-small'
    text_emb_dim = 512
    
if __name__ == "__main__":
    from transformers import T5TokenizerFast

    model = BaseTransformer(BaseTransformerConfig())
    tokenizer = T5TokenizerFast.from_pretrained("t5-small")
    img = torch.randint(0, 1024, (1, 16, 16))
    text = "Monkey showering"
    text_ids = tokenizer(text, return_tensors='pt').input_ids

    print("Testing the forward method with no text input")
    output = model(img, None)
    assert output.shape == torch.Size([1, 16 * 16, 1024])
    print("Testing Passed")

    print("Testing the model with text input")
    output = model(img, text_ids)
    assert output.shape == torch.Size([1, 16 * 16, 1024])
    print("Testing Passed")

    print("Testing the forward method with conditional scale")
    output = model.forward_with_cond_scale(img, text_ids, 0.5)
    assert output.shape == torch.Size([1, 16 * 16, 1024])
    print("Testing Passed")

    super_model = SuperResTransformer(SuperResTransformerConfig())
    hi_res = torch.randint(0, 1024, (1, 64, 64))

    print("Testing the forward method of the super res transformer with no text ")
    output = super_model(img, hi_res)
    assert output.shape == torch.Size([1, 64, 64, 1024])
    print("Testing Passed")

    print("Testing the forward method of the super res transformer with text")
    output = super_model(img, hi_res, text_ids)
    assert output.shape == torch.Size([1, 64, 64, 1024])
    print("Testing Passed")

    print("Testing the forward with conditioning scale method for the super res transformer")
    output = super_model.forward_with_cond_scale(img, hi_res, text_ids, 0.5)
    print("Testing Passed")