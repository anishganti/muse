import gradio as gr
from PIL import Image
import torch
from muse import PipelineMuse, MaskGiTUViT
from compel import Compel, ReturnedEmbeddingsType

# from swin_ir_2 import load_model, preprocesss_image, postprocess_image


device = "cuda" if torch.cuda.is_available() else "cpu"
# pipe = PipelineMuse.from_pretrained("openMUSE/muse-laiona6-uvit-clip-220k").to(device)

pipe = PipelineMuse.from_pretrained(
    transformer_path="valhalla/research-run",
    text_encoder_path="openMUSE/clip-vit-large-patch14-text-enc",
    vae_path="openMUSE/vqgan-f16-8192-laion",
).to(device)
pipe.transformer = MaskGiTUViT.from_pretrained("valhalla/research-run-finetuned-journeydb", subfolder="ema_model", revision="06bcd6ab6580a2ed3275ddfc17f463b8574457da").to(device)
pipe.tokenizer.pad_token_id = 49407

# sr_model = load_model().to(device)

if device == "cuda":
    pipe.transformer.enable_xformers_memory_efficient_attention()
    pipe.text_encoder.to(torch.float16)
    pipe.transformer.to(torch.float16)


compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder, returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=True, truncate_long_prompts=False)

def infer(prompt, negative, scale=10):
    print("Generating:")

    conditioning, pooled = compel(prompt)
    negative_conditioning, negative_pooled = compel(negative)
    conditioning, negative_conditioning = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])

    images = pipe(
        prompt,
        timesteps=16,
        negative_text=None,
        prompt_embeds=conditioning,
        pooled_embeds=pooled,
        negative_prompt_embeds=negative_conditioning,
        negative_pooled_embeds=negative_pooled,
        guidance_scale=scale,
        num_images_per_prompt=9,
        temperature=(2, 0),
        orig_size=(512, 512),
        crop_coords=(0, 0),
        aesthetic_score=6,
        use_fp16=device == "cuda",
        transformer_seq_len=1024,
        use_tqdm=True,
    )
    print("Done Generating!")
    print("Num Images:", len(images))

    # sr_images = [preprocesss_image(image) for image in images]
    # sr_images = torch.cat(sr_images).to("cuda")
    # with torch.no_grad():
    #     sr_images = sr_model(sr_images)
    #     sr_images = sr_images[..., : 256 * 4, : 256 * 4]
    # sr_images = [postprocess_image(im) for im in sr_images]
    # sr_images = [image.resize((512, 512)) for image in sr_images]
    return images


examples = [
    [
        'A high tech solarpunk utopia in the Amazon rainforest',
        'low quality',
        10,
    ],
    [
        'A pikachu fine dining with a view to the Eiffel Tower',
        'low quality',
        10,
    ],
    [
        'A mecha robot in a favela in expressionist style',
        'low quality, 3d, photorealistic',
        10,
    ],
    [
        'an insect robot preparing a delicious meal',
        'low quality, illustration',
        10,
    ],
    [
        "A small cabin on top of a snowy mountain in the style of Disney, artstation",
        'low quality, ugly',
        10,
    ],
]


css = """
h1 {
  text-align: center;
}

#component-0 {
  max-width: 730px;
  margin: auto;
}
"""

block = gr.Blocks(css=css)

with block:
    gr.Markdown("MUSE is an upcoming fast text2image model.")
    with gr.Group():
        with gr.Row(elem_id="prompt-container").style(mobile_collapse=False, equal_height=True):
            with gr.Column():
                text = gr.Textbox(
                    label="Enter your prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=False,
                )
                negative = gr.Textbox(
                    label="Enter your negative prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your negative prompt",
                    container=False,
                )
            btn = gr.Button("Generate image", scale=0)

        gallery = gr.Gallery(
            label="Generated images", show_label=False,
        ).style(grid=[3])

    with gr.Accordion("Advanced settings", open=False):
        guidance_scale = gr.Slider(
            label="Guidance Scale", minimum=0, maximum=20, value=10, step=0.1
        )

    ex = gr.Examples(examples=examples, fn=infer, inputs=[text, negative, guidance_scale], outputs=gallery, cache_examples=False)
    ex.dataset.headers = [""]

    text.submit(infer, inputs=[text, negative, guidance_scale], outputs=gallery)
    negative.submit(infer, inputs=[text, negative, guidance_scale], outputs=gallery)
    btn.click(infer, inputs=[text, negative, guidance_scale], outputs=gallery)

block.launch()
