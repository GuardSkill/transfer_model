from diffusers import StableDiffusionXLPipeline
import torch

ckpt_file = '/SDModel/Stable-diffusion/SDXL/sd_xl_base_1.0.safetensors'
save_folder = './ckpt/sd_xl_base_1.0.safetensors'
pipeline = StableDiffusionXLPipeline.from_single_file(ckpt_file, torch_dtype=torch.bfloat16)
pipeline.save_pretrained(save_folder, safe_serialization=True)

