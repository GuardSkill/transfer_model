from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel, CLIPTextModel
# from optimum.quanto import freeze, qfloat8, quantize
from safetensors.torch import load_file
import torch
import os 
from lycoris import create_lycoris_from_weights
import json 
from lycoris import create_lycoris
try:
    from lycoris import LycorisNetwork
except:
    print("[ERROR] Lycoris not available. Please install ")
    
# merge lora and save
def load_lora():
    dtype = torch.bfloat16
    # transformer = FluxTransformer2DModel.from_single_file("/Disk1/SimpleTuner/ckpt/flux-schnell_adapter", torch_dtype=dtype)
    # bfl_repo = "black-forest-labs/FLUX.1-dev"
    # text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype)

    bfl_repo ="/Disk1/SimpleTuner/ckpt/flux-adapter"
    # bfl_repo ="/Disk1/SimpleTuner/ckpt/flux_dev"
    # pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=None, text_encoder_2=None, torch_dtype=dtype)
    pipe = FluxPipeline.from_pretrained(bfl_repo, torch_dtype=dtype)
    # pipe.transformer = transformer
    # pipe.text_encoder_2 = text_encoder_2
    lora_path="/Disk1/SimpleTuner/output/model_lora_adapter_generated_add_pixabay_low_rate/checkpoint-5100/pytorch_lora_weights.safetensors"
    # lora_path="/Disk1/SimpleTuner/output/model_lora_base_adapter_data_GCI/checkpoint-11000/pytorch_lora_weights.safetensors"
    # state_dict = load_file(lora_path)

    pipe.load_lora_weights(lora_path, adapter_name="feng")
    # pipe.load_lora_weights(os.path.dirname(lora_path), weight_name="pytorch_lora_weights.safetensors", adapter_name="feng")
    pipe.fuse_lora(adapter_names=["feng"], lora_scale=1.0)

    pipe.unload_lora_weights()
    pipe.save_pretrained("/Disk1/SimpleTuner/ckpt/flux-adapter")

# merge lycoris and save
def load_lycoris(lora_path="/Disk1/SimpleTuner/output/model_lora_base_adapter_data_GCI/checkpoint-11000/pytorch_lora_weights.safetensors"):
    dtype = torch.bfloat16
    bfl_repo ="/Disk1/SimpleTuner/ckpt/flux-adapter"
    pipe = FluxPipeline.from_pretrained(bfl_repo, torch_dtype=dtype)
    
    lycoris_config=os.path.dirname(lora_path) + "/lycoris_config.json"
    
    with open(lycoris_config, "r") as f:
        lycoris_config = json.load(f)
    multiplier = int(lycoris_config["multiplier"])
    linear_dim = int(lycoris_config["linear_dim"])
    linear_alpha = int(lycoris_config["linear_alpha"])
    apply_preset = lycoris_config.get("apply_preset", None)
    if apply_preset is not None and apply_preset != {}:
        LycorisNetwork.apply_preset(apply_preset)

    # Remove the positional arguments we extracted.
    del lycoris_config["multiplier"]
    del lycoris_config["linear_dim"]
    del lycoris_config["linear_alpha"]

    print(f"Using lycoris training mode")

    if pipe.transformer is not None:
        model_for_lycoris_wrap = pipe.transformer


    lycoris_net = create_lycoris(
        model_for_lycoris_wrap,
        multiplier,
        linear_dim,
        linear_alpha,
        **lycoris_config,
    )
    lycoris_net.apply_to()
    
    pipe.transformer=model_for_lycoris_wrap
    pipe.save_pretrained("/Disk1/SimpleTuner/ckpt/flux-adapter2",safe_serialization=True, use_safetensors=True)
    
load_lycoris(lora_path="/Disk1/SimpleTuner/output/model_lora_base_adapter_data_GCI/checkpoint-11000/pytorch_lora_weights.safetensors")