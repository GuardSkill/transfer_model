import argparse
import torch
from safetensors.torch import save_file
from safetensors import safe_open
import os

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('old_key_file', type=str, help='Path to the text file containing old keys')
parser.add_argument('new_key_file', type=str, help='Path to the text file containing new keys')
parser.add_argument('old_model', type=str, help='Path to the model containing old keys')
parser.add_argument('new_model', type=str, help='Save path for the renamed model')
args = parser.parse_args()

# Function to convert .pt model to safetensors format
def convert_pt_to_safetensors(pt_path, safe_path):
    model = torch.load(pt_path)  # Load the PyTorch model
    save_file(model, safe_path)  # Save it in safetensors format

# Flag to track if conversion occurs
conversion_performed = False

# Check if the old_model is a .pt file and convert if necessary
model_path = args.old_model
if model_path.endswith('.pt'):
    temp_safe_path = model_path + ".safetensors"  # Temporary safetensors path
    convert_pt_to_safetensors(model_path, temp_safe_path)
    model_path = temp_safe_path  # Use the converted model for further processing
    conversion_performed = True  # Mark that conversion was performed

# Read the keys from the text files
with open(args.new_key_file, 'r') as f:
    new_keys = f.read().splitlines()
with open(args.old_key_file, 'r') as f:
    old_keys = f.read().splitlines()

# Open the safetensors file
with safe_open(model_path, framework="pt") as f:
    # Load all tensors
    tensors = {k: f.get_tensor(k) for k in f.keys()}

    # Rename the keys as needed
    renamed_tensors = {new_key: tensors[old_key] for old_key, new_key in zip(old_keys, new_keys)}
    metadata = {"format":"pt"}

    # Save the renamed tensors back to a safetensors file
    save_file(renamed_tensors, args.new_model, metadata)