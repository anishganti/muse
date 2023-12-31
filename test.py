import torch
from transformers import T5EncoderModel
from model.transformer import BaseTransformer, SuperResTransformer
from configs.config import BaseTransformerConfig, BaseSuperResConfig
from utils import *

# base transformer model
model = BaseTransformer(BaseTransformerConfig())
txt_encoder = T5EncoderModel.from_pretrained('google/t5-efficient-base')
super_model = SuperResTransformer(BaseSuperResConfig())
tokenizer = T5TokenizerFast.from_pretrained("google/t5-efficient-base")

if __name__ == "__main__":

    loader, l = prepare_dataloader('base', 256, 16, 2)
    #super_loader = prepare_dataloader('base', 512, 8, 1)
    #super_sample = next(iter(super_loader))
    sample = next(iter(loader))

    print("Testing the forward method with no text input")
    img_tokens = mask_tokens(sample['image_tokens'])
    output = model(img_tokens, None)
    loss = torch.nn.functional.cross_entropy(output.view(-1, output.size(-1)), sample['image_tokens'].view(-1))
    print(loss)
    assert output.shape == torch.Size([2, 256, 8193])
    print("Testing Passed")

    """ print("Testing the model with text input")
    output = model(sample['image_tokens'], txt_encoder(sample['text_tokens']).last_hidden_state)
    assert output.shape == torch.Size([1, 256, 8193])
    print("Testing Passed")

    print("Testing the forward method with conditional scale")
    output = model.forward_with_cond_scale(sample['image_tokens'], txt_encoder(sample['text_tokens']).last_hidden_state, 0.5)
    assert output.shape == torch.Size([1, 256, 8193])
    print("Testing Passed")

    print("Testing the forward method of the super res transformer with no text")
    output = super_model(sample['image_tokens'], super_sample['image_tokens'])
    assert output.shape == torch.Size([1, 4096, 8193])
    print("Testing Passed")

    print("Testing the forward method of the super res transformer with text")
    output = super_model(sample['image_tokens'], super_sample['image_tokens'], txt_encoder(sample['text_tokens']).last_hidden_state)
    assert output.shape == torch.Size([1, 4096, 8193])
    print("Testing Passed")

    print("Testing the forward with conditioning scale method for the super res transformer")
    output = super_model.forward_with_cond_scale(sample['image_tokens'], super_sample['image_tokens'], txt_encoder(sample['text_tokens']).last_hidden_state, 0.5)
    assert output.shape == torch.Size([1, 4096, 8193])
    print("Testing Passed")  """  
