from modules import *
from random import random
from math import pi

"""
Transformer components for the MUSE model
"""

class SuperResTransformer(nn.Module):
    def __init__(self, config) -> None:
        super(SuperResTransformer, self).__init__()

        self.src_seq_len = config.src_height * config.src_width
        self.tgt_seq_len = config.tgt_height * config.tgt_height
        
        self.transformer = nn.ModuleDict(dict(
            src_embedding = nn.Embedding(config.src_codebook_sz, config.emb_dim),
            tgt_embedding = nn.Embedding(config.tgt_codebook_sz, config.emb_dim),
            src_pos_enc2d = nn.Embedding(config.src_height * config.src_width, config.emb_dim),
            tgt_pos_enc2d = nn.Embedding(config.tgt_height * config.tgt_width, config.emb_dim),
            encoder = nn.ModuleList(
                [EncoderBlock(config.emb_dim, config.nhead, config.dropout) for _ in range(config.n_enc_layer)]
            ),
            decoder = nn.ModuleList(
                [MultiAxisTransformerBlock(config.emb_dim, config.block_size, config.nhead, config.dropout) for _ in range(config.n_dec_layer)],
            ),
            ln_f = nn.LayerNorm(config.emb_dim)
        ))
        # other components of the transformer model
        self.dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.emb_dim, config.tgt_codebook_sz)
        # the text encoder components from t5
        self.text_encoder = get_text_encoder(config.t5_model)
        self.proj_text = nn.Linear(config.text_emb_dim, config.emb_dim)

    def forward(self, low_res, hi_res, text):
        n, m, device = self.src_seq_len, self.tgt_seq_len, low_res.device
        # encode and project the text into our model's dimension
        text_emb = self.text_encoder(text)
        text_emb = self.proj_text(text_emb.last_hidden_states)
        # embed the image and inject positional information
        src_emb = self.transformer.src_embedding(low_res)
        src_emb = self.dropout(src_emb + self.transformer.src_pos_enc2d(torch.arange(n, device=device)))
        # pass the source embedding to encoders
        src_emb = self.transformer.encoder(src_emb)
        # concat the low-res images with the text embedding
        src_emb = torch.cat(src_emb, text_emb, dim=-1)
        # embed the hi-res tokens
        tgt_emb = self.transformer.tgt_embedding(hi_res)
        tgt_emb = self.dropout(tgt_emb + self.transformer.tgt_pos_enc2d(torch.arange(m, device=device)))
        # pass the target embedding to the transformers
        tgt_emb = self.transformer.decoder(tgt_emb, src_emb)
        tgt_emb = self.transformer.ln_f(tgt_emb)

        return self.lm_head(tgt_emb)

class BaseTransformer(nn.Module):
    def __init__(self, config) -> None:
        super(BaseTransformer, self).__init__()
        # save the sequence length of the base images
        self.height = config.height
        self.width = config.width
        self.seq_len = config.height * config.width      
        # create the transformer model
        self.transformer = nn.ModuleDict(dict(
            img_embedding = nn.Embedding(config.codebook_sz, config.emb_dim),
            pos_enc2d = nn.Embedding(config.height * config.width, config.emb_dim),
            decoder = nn.ModuleList(
                [TransformerBlock(config.emb_dim, config.nhead, config.dropout) for _ in range(config.n_layer)]
            ),
            ln_f = nn.LayerNorm(config.emb_dim)
        ))
        # other components of the transformer model
        self.dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.emb_dim, config.codebook_sz)
        # the text encoder components from t5
        self.text_encoder = get_text_encoder(config.t5_model)
        self.proj_text = nn.Linear(config.text_emb_dim, config.emb_dim)

    def get_mask(self):
        p = 2/pi * (1 - random.random() ** 2) ** (-1/2)
        mask = (torch.randn(self.seq_len) < p)
        self.register_buffer("mask_tokens", mask, persistent=False)
        return mask

    def forward(self, img, text, mask_id:int):
        n, device = self.seq_len, img.device
        # encode and project the text into our model's dimension
        # only do this 90% of the time
        if text is not None:
            text_emb = self.text_encoder(text)
            text_emb = self.proj_text(text_emb.last_hidden_states)
        else:
            text_emb = text

        # embed the image and inject positional information
        # mask the tokens based on the muse strategy
        # flatten the image tokens for computation
        img_emb = self.transformer.img_embedding(img.flatten(dim=1).masked_fill(self.get_mask(), mask_id))
        img_emb = self.dropout(img_emb + self.transformer.pos_enc2d(torch.arange(n, device=device)))
        # pass the image embedding through the transformer block
        img_emb = self.transformer.decoder(img_emb, text_emb)
        img_emb = self.transformer.ln_f(img_emb)

        return self.lm_head(img_emb)
    
    def forward_with_cond_scale(self, img, text, mask_id:int, cond_scale:float):
        # entirely conditioned logits
        if cond_scale == 1.:
            return self.forward(img, text, mask_id)
        
        logits = self.forward(img, text, mask_id) # logits conditioned on text
        null_logits = self.forward(img, None, mask_id) # logits with no text conditioning

        # cfg logits
        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        return scaled_logits
    
    def forward_with_neg_prompt(self, img, pos_text, neg_text, mask_id:int, cond_scale:float):
        # logits with negative prompts
        neg_logits = self.forward(img, neg_text, mask_id)
        # logits with positive prompts
        pos_logits = self.forward(img, pos_text, mask_id)

        scaled_logits = neg_logits + (pos_logits - neg_logits) * cond_scale

        return scaled_logits