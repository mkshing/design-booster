import os
import io
import requests
from PIL import Image
from torchvision import transforms as T
import torch
from torch import autocast
from diffusers import UNet2DConditionModel
from design_booster.modeling_designbooster import DesignBoosterModel
from design_booster.pipeline_designbooster import StableDiffusionDesignBoosterPipeline


def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


model_path = "CKPT_MODEL_PATH" # @param {type: "string"}
BASE_MODEL_NAME = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionDesignBoosterPipeline.from_pretrained(
    BASE_MODEL_NAME,
    safety_checker=None,
    unet=UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", use_auth_token=True),
    text_encoder=DesignBoosterModel.from_pretrained(model_path, subfolder="text_encoder", use_auth_token=True),
    torch_dtype=torch.float16,
).to("cuda")
pipe.enable_xformers_memory_efficient_attention()


# prompt
prompt = "" #@param {type:"string"}
image_path_or_url = "/content/test.jpeg" # @param {type:"string"}
# negative_prompt = "" #@param {type:"string"}
num_samples = 2 # @param {type: "integer"}
guidance_scale = 10 # @param {type: "number"}
num_inference_steps = 100 # @param {type: "integer"}
sigma_switch_step = 30 # @param {type: "integer"}
output_dir = "/content/txt2img-outputs/" #@param {type:"string"}
os.makedirs(output_dir, exist_ok=True)

init_image = Image.open(fetch(image_path_or_url)).convert("RGB")
resize_fn = T.Compose([
    T.Resize(size=512, interpolation=T.InterpolationMode.LANCZOS),
    T.CenterCrop(size=512)
])
init_image = resize_fn(init_image)

with autocast("cuda"), torch.inference_mode():
    images = pipe(
        prompt,
        image=init_image,
        sigma_switch_step=sigma_switch_step,
        num_images_per_prompt=num_samples,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images

for i, img in enumerate(images):
    img.save(f"{output_dir}{i}.png")
