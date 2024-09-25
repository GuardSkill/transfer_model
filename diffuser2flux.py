from safetensors.torch import load_file, save_file 
from transformers import AutoTokenizer, AutoModel 
from diffusers import StableDiffusionPipeline 
import torch
import glob

path="/Disk1/SimpleTuner/ckpt/flux-adapterV03/transformer/diffusion_pytorch_model-*.safetensors"
all_paths = glob.glob(path)

merged_state_dict = {}

for path in all_paths:
    loaded_dict = load_file(path)
    merged_state_dict.update(loaded_dict)

# Save the merged dictionary
save_file(merged_state_dict, "merged_diffusion_torch.safetensors")
