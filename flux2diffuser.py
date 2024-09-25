from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel, CLIPTextModel
# from optimum.quanto import freeze, qfloat8, quantize
import torch

model_path = "/Disk1/ComfyUI/models/unet/flux-merged_dev_schnell_.safetensors" # Your authenticated HF token
model_path = "/Disk1/ComfyUI/models/unet/dev_schnell_pixabay_00001_.safetensors"  # dev+shnell+pixabay
model_path = "/Disk1/Models/V3/Flux_schnell_adapter.safetensors"
model_path = "/Disk1/Models/V3/v03_adapter.safetensors"
transformer = FluxTransformer2DModel.from_single_file(model_path, torch_dtype = torch.bfloat16)  # get 3 safetensormodel # some time need HF_ENDPOINT=https://hf-mirror.com python flux2diffuser.py
transformer.save_pretrained("/Disk1/SimpleTuner/ckpt/flux-adapterV03", safe_serialization=True, use_safetensors=True)
