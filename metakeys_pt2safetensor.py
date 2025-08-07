import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import save_file
import argparse
import os
import tempfile

    
# Function to convert .pt model to safetensors format
def convert_pt_to_safetensors(pt_path, updated_path, safe_path):
    # unsafe_globals = []
    # unsafe_globals = torch.serialization.get_unsafe_globals_in_checkpoint(pt_path)
    # torch.serialization.add_safe_globals(unsafe_globals)
    model = torch.load(pt_path, map_location="cpu", weights_only=False)  # Load the PyTorch model
    print(model)
    torch.save(model, updated_path)
    model = torch.load(updated_path, map_location="cpu")
    metadata = {"format":"pt"}
    save_file(model, safe_path, metadata)  # Save it in safetensors format

def load_metadata_from_safetensors(safetensors_file: str) -> dict:
    """
    This method locks the file. see https://github.com/huggingface/safetensors/issues/164
    If the file isn't .safetensors or doesn't have metadata, return empty dict.
    """
    with safe_open(safetensors_file, framework="pt", device="cpu") as f:
        model_weights = {}
        for key in f.keys():
            model_weights[key] = f.get_tensor(key)
        metadata = f.metadata()
    if metadata is None:
        metadata = {}
    return model_weights, metadata

def get_shapes(model_path):
    """Gets layer name and shapes from PyTorch model loaded from safetensors file outputs it to file.

    Args:
        model_a_path (str): Path to the safetensors file of the model.
        output_path (str): Path to the text file where results will be written.
    """

    # Derive output path from model path
    base_name, _ = os.path.splitext(model_path)
    output_path = f"{base_name}_metakeys.txt"
    index = 1
    
    while os.path.exists(output_path):  # Check for existing files
        output_path = f"{base_name}_metakeys_{index}.txt"
        index += 1

    # Load model weights from safetensors file
    model_weights = {}
    metadata = {}
    model_weights, metadata = load_metadata_from_safetensors(model_path)



    # Write results to the output file
    with open(output_path, "w") as f:
        f.write(f"{metadata}\n")
        for layer_name in model_weights.keys():
            tensor_a = model_weights[layer_name]
            shape_a = str(model_weights[layer_name].shape)
            shape_a = shape_a.replace("torch.Size(", "").replace(")", "")
            f.write(f"{layer_name}, {shape_a}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gets layer names and shapes of PyTorch model.")
    parser.add_argument("model_path", type=str, help="Path to the safetensors file of the model")


    args = parser.parse_args()

    # Flag to track if conversion occurs
    conversion_performed = False

    # Check if the old_model is a .pt file and convert if necessary
    model_path = args.model_path
    if model_path.endswith('.pt') or model_path.endswith('.pth') or model_path.endswith('.bin') or model_path.endswith('.ckpt'):
        temp_safe_path = model_path + ".safetensors"  # Temporary safetensors path
        temp_updated_path = model_path + "_temp.pt"
        convert_pt_to_safetensors(model_path, temp_updated_path, temp_safe_path)
        model_path = temp_safe_path  # Use the converted model for further processing
        conversion_performed = True  # Mark that conversion was performed

    get_shapes(model_path)

    # if conversion_performed:
    #     os.remove(model_path)