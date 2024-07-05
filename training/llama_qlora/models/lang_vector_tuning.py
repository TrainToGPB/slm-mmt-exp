import os
from tqdm import tqdm

import torch
import GPUtil
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


def check_gpu(vram=True, util=False):
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU ID {gpu.id}: {gpu.name}")
        if vram:
            vram_used = gpu.memoryUsed / 1024
            vram_max = gpu.memoryTotal / 1024
            vram_percent = (vram_used / vram_max) * 100
            print(f"  - VRAM: {vram_used:.2f} GB / {vram_max:.2f} GB ({vram_percent:.2f}%)")
        if util:
            gpu_util = gpu.load * 100
            gpu_temp = gpu.temperature
            print(f"  - Utilization: {gpu_util}%")
            print(f"  - Temperature: {gpu_temp}°C")


def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 torch_dtype=torch.bfloat16, 
                                                 cache_dir='/data/.cache/hub', 
                                                 device_map='auto')
    model = model.to(DEVICE)
    return model


def normalize_tensor(tensor, epsilon=1e-6):
    tensor_min, tensor_max = tensor.min().item(), tensor.max().item()
    if abs(tensor_max - tensor_min) < epsilon:
        normalized_tensor = torch.zeros_like(tensor)
    else:
        normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    return normalized_tensor


def get_ratio(diff_vector, method='sigmoid', epsilon=1e-6, **kwargs):
    if isinstance(diff_vector, str):
        diff_vector = load_model(diff_vector)

    if method not in ['sigmoid', 'linear']:
        raise ValueError(f"Invalid scaling method: {method}")

    diff_dict = diff_vector.state_dict() if isinstance(diff_vector, torch.nn.Module) else diff_vector

    ratio_dict = {}
    tqdm_iterator = tqdm(diff_dict.keys(), total=len(list(diff_dict.keys())), desc="Calculating ratio")
    for key in tqdm_iterator:
        diff_tensor = abs(diff_dict[key])
        diff_norm = normalize_tensor(diff_tensor, epsilon=epsilon)

        if method == 'sigmoid':
            scale_factor, shift_factor = kwargs.get('sigmoid_scale_factor', 12.0), kwargs.get('sigmoid_shift_factor', 6.0)
            diff_sigmoid = torch.sigmoid(diff_norm * scale_factor - shift_factor)
            ratio_dict[key] = 1 - diff_sigmoid

        elif method == 'linear':
            ratio_dict[key] = 1 - diff_norm

        ratio_dict[key] = ratio_dict[key].to(DEVICE)

    return ratio_dict


def add_models(model1, model2, model_name=None, **kwargs):
    if isinstance(model1, dict) and isinstance(model2, dict) and model_name is None:
        raise ValueError("\'model_name\' is required when both models are vectors in dictionary.")

    if isinstance(model1, str):
        model1 = load_model(model1)
    if isinstance(model2, str):
        model2 = load_model(model2)
    
    model_name = model1 if isinstance(model1, str) else model1.config.name_or_path

    if isinstance(model1, dict):
        model1 = transform_dict_to_model(model1, model_name)
    if isinstance(model2, dict):
        model2 = transform_dict_to_model(model2, model_name)

    a, b = kwargs.get('a', 1.0), kwargs.get('b', 1.0)

    model_sum = {}
    model1_dict = model1.state_dict()
    model2_dict = model2.state_dict()

    tqdm_iterator = tqdm(model1_dict.keys(), total=len(list(model1_dict.keys())), desc="Adding models")
    for key in tqdm_iterator:
        if key in model2_dict:
            if isinstance(b, dict) and key in b:
                try:
                    model_sum[key] = (a * model1_dict[key]) + (b[key] * model2_dict[key])
                except RuntimeError:
                    print(f"model1_dict[{key}] -> device: {model1_dict[key].device}")
                    print(f"model2_dict[{key}] -> device: {model2_dict[key].device}")
                    print(f"ratio[{key}] -> device: {b[key].device}")
                    raise RuntimeError('Different devices')
            elif isinstance(b, (int, float)):
                model_sum[key] = (a * model1_dict[key]) + (b * model2_dict[key])
            else:
                raise ValueError(f"Invalid value for \'b\': {b}")
            
    model_sum = transform_dict_to_model(model_sum, model_name)

    return model_sum


def transform_dict_to_model(state_dict, backbone_name):
    config = AutoConfig.from_pretrained(backbone_name)
    model = AutoModelForCausalLM.from_config(config)

    model.load_state_dict(state_dict)

    model = model.to(torch.bfloat16)
    model = model.to(DEVICE)

    return model


def save_model(model, output_dir, backbone_name=None):
    if isinstance(model, dict):
        if backbone_name is None:
            raise ValueError("\'backbone_name\' is required when model is a dictionary.")
        model = transform_dict_to_model(model, backbone_name)

    training_args = TrainingArguments(output_dir)
    trainer = Trainer(model=model, args=training_args)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)


def calculate_frobenius_norm(model1, model2=None, add_vector=False, print_result=True):
    if isinstance(model1, str):
        model1 = load_model(model1)
    if model2 is not None and isinstance(model2, str):
        model2 = load_model(model2)

    if model2:
        b = 1.0 if add_vector else -1.0
        model_diff = add_models(model1, model2, a=1.0, b=b)
    else:
        model_diff = model1

    frobenius_norm = sum(
        torch.norm(param, p='fro').item() for param in model_diff.parameters()
    )
    
    if print_result:
        print(f"Frobenius norm: {frobenius_norm:.3f}")

    return frobenius_norm


def combine_all_weights(model):
    if isinstance(model, str):
        model = load_model(model)
        state_dict = model.state_dict()
    elif isinstance(model, torch.nn.Module):
        state_dict = model.state_dict()
    elif isinstance(model, dict):
        state_dict = model

    all_weights = []
    for key, tensor in tqdm(state_dict.items(), total=len(state_dict), desc="Combining weights"):
        # print(f"Combining {key}...")
        tensor = tensor.to(torch.float16)
        all_weights.append(tensor.cpu().numpy().flatten())
    combined_weights = np.concatenate(all_weights)
    
    return combined_weights


def weight_statistics(model):
    combined_weights = combine_all_weights(model)
    
    print("\n[WEIGHT STATISTICS]")

    mean = np.mean(combined_weights)
    print(f"  - mean: {mean:.7f}")
    median = np.median(combined_weights)
    print(f"  - median: {median:.7f}")
    std = np.std(combined_weights)
    print(f"  - std: {std:.7f}")
    
    min_val = np.min(combined_weights)
    print(f"  - min: {min_val:.7f}")
    max_val = np.max(combined_weights)
    print(f"  - max: {max_val:.7f}")


def weight_histogram(model, bins=100):
    combined_weights = combine_all_weights(model)
    plt.hist(combined_weights, bins=bins)
    plt.title("Histogram of All Weights")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    model_name = 'meta-llama/Meta-Llama-3-8B'
    vector1_name = 'traintogpb/llama-3-lang-vector-ko'
    vector2_name = 'traintogpb/llama-3-lang-vector-ja'
    vector3_name = 'traintogpb/llama-3-lang-vector-zh'

    vector1 = load_model(vector1_name)
    vector2 = load_model(vector2_name)

    # ratio = get_ratio(vector1, method='linear', sigmoid_scale_factor=6.0, sigmoid_shift_factor=3.0)

    # weight_statistics(ratio)

    vector_mix = add_models(vector1, vector2, a=1.0, b=1.0)

    del vector1
    del vector2
    torch.cuda.empty_cache()

    vector3 = load_model(vector3_name)
    vector_mix = add_models(vector_mix, vector3, a=1.0, b=1.0)
    del vector3
    torch.cuda.empty_cache()

    model = load_model(model_name)
    model_multiling = add_models(model, vector_mix, a=1.0, b=1.0)
    del model
    torch.cuda.empty_cache()

    calculate_frobenius_norm(vector_mix, add_vector=False)

    del vector_mix
    torch.cuda.empty_cache()

    save_model(model_multiling, '/data/sehyeong/nmt/models/langvec_plm/llama3-multilingual-kojazh-langvec')


if __name__ == '__main__':
    main()
