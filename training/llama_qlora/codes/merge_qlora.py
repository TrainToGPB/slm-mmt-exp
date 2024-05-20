"""
This script is used to merge the adapter with the dequantized model and save it to the specified directory

The following functions are available:
- save_model: Save the model to the specified directory
- dequantize_model: Dequantize the model and save it to the specified directory
- dequantize_and_save: Dequantize the model and save it to the specified directory
- convert_bf16_to_fp16: Convert the model from bfloat16 to float16 and save it to the specified directory

Example:
    python merge_qlora.py

Notes:
    This script is intended to be used with the OpenAI GPT-3 model and the BitsAndBytes library
    The model is dequantized and merged with the adapter
    The dequantized model is saved to the specified directory
    The merged model is saved to the specified directory
    The model is converted from bfloat16 to float16 and saved to the specified directory
"""
# built-in
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import gc
import json
import copy

# third-party
import torch
import shutil
from tqdm import tqdm
from peft import PeftModel
from peft.utils import _get_submodules
import bitsandbytes as bnb
from bitsandbytes.functional import dequantize_4bit
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig


def save_model(model, tokenizer, to):
    """
    Save the model to the specified directory

    Args:
    - model: The model to be saved
    - tokenizer: The tokenizer to be saved
    - to: The directory to save the model
    """
    print(f"Saving dequantized model to {to}...")
    model.save_pretrained(to)
    tokenizer.save_pretrained(to)
    config_data = json.loads(open(os.path.join(to, 'config.json'), 'r').read())
    config_data.pop("quantization_config", None)
    config_data.pop("pretraining_tp", None)
    with open(os.path.join(to, 'config.json'), 'w') as config:
        config.write(json.dumps(config_data, indent=2))
    

def dequantize_model(model, tokenizer, to=None, dtype=torch.bfloat16, device=torch.cuda.current_device()):
    """
    Dequantize the model and save it to the specified directory

    Args:
    - model: The model to be dequantized
    - tokenizer: The tokenizer to be saved
    - to: The directory to save the model
    - dtype: The data type to convert the model to
    - device: The device to move the model to

    Returns:
    - The dequantized model
    """
    cls = bnb.nn.Linear4bit

    with torch.no_grad():
        tqdm_iterator = tqdm(model.named_modules(), total=len(list(model.named_modules())))
        tqdm_iterator.set_description('[DEQUANTIZING LAYERS]')
        for name, module in tqdm_iterator:
            if isinstance(module, cls):
                quant_state = copy.deepcopy(module.weight.quant_state)
                quant_state.dtype = dtype

                weights = dequantize_4bit(module.weight.data, quant_state=quant_state, quant_type="nf4").to(dtype)

                new_module = torch.nn.Linear(module.in_features, module.out_features, bias=None, dtype=dtype)
                new_module.weight = torch.nn.Parameter(weights)
                new_module.to(device=device, dtype=dtype)

                parent, target, target_name = _get_submodules(model, name)
                setattr(parent, target_name, new_module)

                print(f"{name} is dequantized...")

        model.is_loaded_in_4bit = False

        if to is not None:
            if os.path.exists(to):
                shutil.rmtree(to)

            os.makedirs(to, exist_ok=True)

            save_model(model, tokenizer, to)
        
        return model
        

def dequantize_and_save(
        model_path,
        adapter_path,
        save_dequant_plm_path,
        save_dequant_merged_path
    ):
    """
    Dequantize the model and save it to the specified directory

    Args:
    - model_path: The path to the model to be dequantized
    - adapter_path: The path to the adapter to be merged with the model
    - save_dequant_plm_path: The directory to save the dequantized model
    - save_dequant_merged_path: The directory to save the merged model

    Returns:
    - The dequantized model
    """
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    try:
        print(f"Starting to load the model {model_path} into memory")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map=torch.cuda.current_device()
        )
        print(model)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        model = dequantize_model(
            model, 
            tokenizer, 
            to=save_dequant_plm_path
        )
        
        print(model)
        model = PeftModel.from_pretrained(model, adapter_path)
        print(model)
        model = model.merge_and_unload()
        print(model)
        
        print(f"Successfully loaded the model {model_path} into memory")
        
        save_model(model, tokenizer, save_dequant_merged_path)
        
        print(f"Successfully saved merged model {model_path} to disk")

    except Exception as e:
        print(f"An error occurred: {e}")

        if 'model' in locals():
            del model

        torch.cuda.empty_cache()

        gc.collect()

        print("Model, GPU cache, and garbage have been cleared.")


def convert_bf16_to_fp16(model_path, save_path):
    """
    Convert the model from bfloat16 to float16 and save it to the specified directory

    Args:
    - model_path: The path to the model to be converted
    - save_path: The directory to save the converted model

    Returns:
    - The converted model
    """
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.half()

    for name, param in model.named_parameters():
        print(f'{name}: {param.dtype}')

    save_model(model, tokenizer, save_path)


if __name__ == '__main__':
    model_path = 'beomi/Llama-3-Open-Ko-8B' # KoEn의 경우: beomi/Llama-3-KoEn-8B-preview
    adapter_paths = [
        '../models/llama3-koen-sparta',
        # '../models/llama3-ko-sparta-odd',
        # '../models/llama3-ko-tiny',
        # '../models/llama3-ko-mini',
    ]
    save_dequant_plm_path = '../models/open-llama-3-ko-8b-bf16-dequant-from-nf4' # KoEn의 경우: open-llama-3-koen-8b-bf16-dequant-from-nf4
    save_dequant_merged_bf16_paths = [
        '../models/llama3-koen-sparta-merged-bf16',
        # '../models/llama3-ko-sparta-odd-merged-bf16',
        # '../models/llama3-ko-tiny-merged-bf16',
        # '../models/llama3-ko-mini-merged-bf16',
    ]

    for adapter_path, save_dequant_merged_bf16_path in zip(adapter_paths, save_dequant_merged_bf16_paths):
        dequantize_and_save(model_path, adapter_path, save_dequant_plm_path, save_dequant_merged_bf16_path)

    # convert_bf16_to_fp16(save_dequant_merged_bf16_path, save_dequant_merged_fp16_path)
    