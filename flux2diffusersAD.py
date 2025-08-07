from diffusers import FluxPipeline, FluxTransformer2DModel
import torch

# path_transformer = "https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8.safetensors"
path_transformer = "/SDModel/Stable-diffusion/PicLumen_Schnell_Art_v1.safetensors"
path_transformer = "/Disk1/ComfyUI/output/diffusion_models/transformer_art_libre_merge_00001_.safetensors"

# path_schnell_framwork="black-forest-labs/FLUX.1-schnell"
path_diffuser="/Disk1/SimpleTuner/ckpt/FLUX.1-schnell"
path_diffuser="/Disk1/Models/LibreFlux-SimpleTuner"


saved_path = "/Disk1/SimpleTuner/ckpt/FLUX.1-Art_LibreFlux_merge"
transformer = FluxTransformer2DModel.from_single_file(path_transformer, torch_dtype = torch.bfloat16)
pipe = FluxPipeline.from_pretrained(path_diffuser, transformer=transformer,torch_dtype = torch.bfloat16)
pipe.save_pretrained(saved_path)