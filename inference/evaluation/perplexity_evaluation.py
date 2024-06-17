import math
import random
import argparse
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


def compute_perplexity(model_name, dataset_name, dataset_lang, num_samples=1000):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    print(f"[EOS] {tokenizer.eos_token} ({tokenizer.eos_token_id})")
    print(f"[PAD] {tokenizer.pad_token} ({tokenizer.pad_token_id})")
    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')
    
    random.seed(42)
    dataset = load_dataset(dataset_name, f'unshuffled_deduplicated_{dataset_lang}', cache_dir=f'/data/sehyeong/nmt/datasets/oscar/{dataset_lang}')
    random_idx = random.sample(range(len(dataset['train'])), num_samples)
    dataset = dataset['train'].select(random_idx)
    
    total_loss = 0.0
    total_length = 0
    
    for sample in tqdm(dataset, total=len(dataset), desc="Computing perplexity"):
        inputs = tokenizer(sample['text'], return_tensors='pt', truncation=True, padding=True, max_length=8192)
        if torch.cuda.is_available():
            inputs = {key: value.to('cuda') for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
        
        total_loss += loss.item() * inputs['input_ids'].size(1)
        total_length += inputs['input_ids'].size(1)
        torch.cuda.empty_cache()
    
    avg_loss = total_loss / total_length
    perplexity = math.exp(avg_loss)
    
    return perplexity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute perplexity of a language model on a given dataset.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to evaluate.")
    parser.add_argument("--dataset_name", type=str, default='oscar-corpus/oscar', help="Name of the dataset to use.")
    parser.add_argument("--dataset_lang", type=str, default='ko', help="Split of the dataset to use.")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to evaluate on.")

    args = parser.parse_args()
    
    perplexity = compute_perplexity(args.model_name, args.dataset_name, args.dataset_lang, args.num_samples)
    print(f"MODEL: {args.model_name}")
    print(f"DATASET: {args.dataset_lang}")
    print(f"Perplexity: {perplexity}")