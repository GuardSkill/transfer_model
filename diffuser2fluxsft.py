from safetensors.torch import load_file, save_file 
from transformers import AutoTokenizer, AutoModel 
from diffusers import StableDiffusionPipeline 
import torch
import glob
from diffusers import FluxPipeline, FluxTransformer2DModel
import torch
import os
# method 1  ## only comfyui loaderd


path_diffuser="/Disk1/Models/LibreFlux-SimpleTuner"
saved_file = "/Disk1/Models/LibreFlux-SimpleTuner-transformer.safetensors"

#path_template="/Disk1/SimpleTuner/ckpt/flux-adapterV03/transformer/diffusion_pytorch_model-*.safetensors"
path_template=os.path.join(path_diffuser,"transformer/diffusion_pytorch_model-*.safetensors")

all_paths = glob.glob(path_template)

merged_state_dict = {}

for path in all_paths:
    loaded_dict = load_file(path)
    merged_state_dict.update(loaded_dict)

# Save the merged dictionary
save_file(merged_state_dict, saved_file)




