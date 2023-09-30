from model.modules import *

"""
Transformer components for the MUSE model
"""

class SuperResTransformer(nn.Module):
    def __init__(self, config, base_transformer) -> None:
        super(SuperResTransformer, self).__init__()

        # cache shape information
        self.hi_res_img_size = config.hi_res_img_size
        self.hi_res_seq_len = config.hi_res_img_size ** 2
        self.flat_block_size = config.block_size ** 2
        
        # creating the transformer
        self.transformer = nn.ModuleDict(dict(
            hi_res_embedding = nn.Embedding(config.codebook_size, config.hidden_size),
            hi_res_pos_enc2d = nn.Embedding(config.hi_res_img_size ** 2, config.hidden_size),
            encoder = nn.ModuleList(
                [EncoderBlock(config.hidden_size, config.num_head, config.dropout) for _ in range(config.num_enc_layer)]
            ),
            decoder = nn.ModuleList(
                [MultiAxisTransformerBlock(config.hidden_size, config.block_size, config.num_head, config.dropout) for _ in range(config.num_dec_layer)],
            ),
            ln_f = LayerNorm(config.hidden_size)
        ))

        # base transformer components
        self.base_transformer = base_transformer
        
        # freeze the weights of the base transformer
        for param in self.base_transformer.parameters():
            param.requires_grad = False

        # other components of the transformer model
        self.dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.codebook_size)
        self.proj_text = nn.Linear(config.text_hidden_size, config.hidden_size)
        self.device = config.device

    def forward(self, low_res_tokens, hi_res_tokens, text_embedding=None):

        # encode and project the text into our model's dimension
        if text_embedding is not None:
            text_embedding = self.proj_text(text_embedding)

        # embed the image and inject positional information
        low_res_embedding = self.base_transformer.transformer.img_embedding(low_res_tokens)
        low_res_embedding = self.dropout(
            low_res_embedding + self.base_transformer.transformer.pos_enc2d(torch.arange(self.base_transformer.seq_len, device=self.device))
        )
        
        # pass the source embedding to encoders
        for enc in self.transformer.encoder:
            low_res_embedding = enc(low_res_embedding)

        # concat the low-res images with the text embedding only if the text is not none
        if text_embedding is not None:
            low_res_embedding = torch.cat((low_res_embedding, text_embedding), dim=-2)

        # embed the hi-res tokens
        hi_res_embedding = self.transformer.hi_res_embedding(hi_res_tokens)
        hi_res_embedding = self.dropout(
            hi_res_embedding + self.transformer.hi_res_pos_enc2d(torch.arange(self.hi_res_seq_len, device=self.device))
        )

        hi_res_embedding = hi_res_embedding.unfold(1, self.flat_block_size, self.flat_block_size).transpose(-2, -1)

        # pass the target embedding to the transformers
        for dec in self.transformer.decoder:
            hi_res_embedding = dec(hi_res_embedding, low_res_embedding)

        hi_res_embedding = hi_res_embedding.flatten(start_dim=1, end_dim=2)
        hi_res_embedding = self.transformer.ln_f(hi_res_embedding)

        return self.lm_head(hi_res_embedding)
    
    def forward_with_cond_scale(self, low_res_tokens, hi_res_tokens, text_embedding, cond_scale:float):
        # entirely conditioned logits
        if cond_scale == 1.:
            return self.forward(low_res_tokens, hi_res_tokens, text_embedding)
        
        logits = self.forward(low_res_tokens, hi_res_tokens, text_embedding) # logits conditioned on text
        null_logits = self.forward(low_res_tokens, hi_res_tokens, text_embedding=None) # logits with no text conditioning

        # cfg logits
        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        return scaled_logits

    def forward_with_neg_prompt(self, low_res_tokens, hi_res_tokens, positive_text_embedding, negative_text_embedding, cond_scale:float):
        # logits with negative prompts
        neg_logits = self.forward(low_res_tokens, hi_res_tokens, negative_text_embedding)
        # logits with positive prompts
        pos_logits = self.forward(low_res_tokens, hi_res_tokens, positive_text_embedding)

        scaled_logits = neg_logits + (pos_logits - neg_logits) * cond_scale

        return scaled_logits

class BaseTransformer(nn.Module):
    def __init__(self, config) -> None:
        super(BaseTransformer, self).__init__()

        # save the sequence length of the base images
        self.seq_len = config.img_size ** 2

        # create the transformer model
        self.transformer = nn.ModuleDict(dict(
            img_embedding = nn.Embedding(config.codebook_size, config.hidden_size),
            pos_enc2d = nn.Embedding(self.seq_len, config.hidden_size),
            decoder = nn.ModuleList(
                [TransformerBlock(config.hidden_size, config.num_head, config.dropout) for _ in range(config.num_layer)]
            ),
            ln_f = LayerNorm(config.hidden_size)
        ))
        # other components of the transformer model
        self.dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.codebook_size)
        self.proj_text = nn.Linear(config.text_hidden_size, config.hidden_size)
        self.device = config.device

    def forward(self, img_tokens, text_embedding=None):

        # encode and project the text into our model's dimension
        # only do this 90% of the time
        if text_embedding is not None:
            text_embedding = self.proj_text(text_embedding)

        # embed the image and inject positional information
        img_embedding = self.transformer.img_embedding(img_tokens)
        img_embedding = self.dropout(img_embedding + self.transformer.pos_enc2d(torch.arange(self.seq_len, device=self.device)))

        # pass the image embedding through the transformer block
        for block in self.transformer.decoder:
            img_emb = block(img_embedding, text_embedding)

        img_emb = self.transformer.ln_f(img_embedding)

        return self.lm_head(img_embedding)
    
    def forward_with_cond_scale(self, img_tokens, text_embedding, cond_scale:float):
        # entirely conditioned logits
        if cond_scale == 1.:
            return self.forward(img_tokens, text_embedding)
        
        logits = self.forward(img_tokens, text_embedding) # logits conditioned on text
        null_logits = self.forward(img_tokens, None) # logits with no text conditioning

        # cfg logits
        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        return scaled_logits
    
    def forward_with_neg_prompt(self, img_tokens, positive_text_embedding, negative_text_embedding, cond_scale:float):

        # logits with negative prompts
        neg_logits = self.forward(img_tokens, negative_text_embedding)

        # logits with positive prompts
        pos_logits = self.forward(img_tokens, positive_text_embedding)

        scaled_logits = neg_logits + (pos_logits - neg_logits) * cond_scale

        return scaled_logits