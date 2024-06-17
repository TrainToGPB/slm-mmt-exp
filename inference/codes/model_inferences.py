"""
Perform inference on the specified dataset using the specified pre-trained language model.

The following models are supported:
- llama
- llama-qlora
- llama-qlora-bf16 (merged & upscaled)
- llama-qlora-fp16 (merged & upscaled)
- llama-qlora-bf16-vllm (merged & upscaled + vLLM)

The following datasets are supported:
- aihub: AI Hub integrated dataset (ref: https://huggingface.co/datasets/traintogpb/aihub-koen-translation-integrated-tiny-100k)
- flores: FLoRes-101 dataset (ref: https://huggingface.co/datasets/gsarti/flores_101)

CLI example:
- Inference on the AI Hub dataset:
    $ python model_inferences.py --model_type=mbart --inference_type=dataset --dataset=aihub
- Inference on a single sentence:
    $ python model_inferences.py --model_type=llama-bf16-vllm --inference_type=sentence --sentence="Hello, world!"

Output:
- Translated dataset file (CSV format)
- Translated sentence (print)

Notes:
- The translated dataset file will be saved in the same directory as the original dataset file.
- The translated sentence will be printed on the console.
"""
# built-in
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['VLLM_USE_MODELSCOPE'] = 'false'
import re
import sys
import argparse
from tqdm import tqdm
from datetime import datetime

# third-party
import torch
import pandas as pd
from peft import PeftModel
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import MarianMTModel, MarianTokenizer
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from transformers import M2M100ForConditionalGeneration, NllbTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from transformers import BitsAndBytesConfig

# custom
sys.path.append(os.path.join(SCRIPT_DIR, '../../'))
from training.training_utils import set_seed


LANG_TABLE = {
    "en": "English",
    "ko": "한국어"
}
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)
TENSOR_PARALLEL_SIZE = 1


def load_model_and_tokenizer(model_type):
    """
    Load pre-trained language model and tokenizer based on the model type.

    Parameters:
    - model_type (str): Type of the pre-trained language model.

    Returns:
    - model (PreTrainedModel): Pre-trained language model.
    - tokenizer (PreTrainedTokenizer): Pre-trained tokenizer.
    """
    # Model mapping
    model_mapping = {
        'llama-2-vllm': ('meta-llama/Llama-2-7b-hf', None, LlamaTokenizer),
        'llama-2-chat-vllm': ('meta-llama/Llama-2-7b-chat-hf', None, LlamaTokenizer),
        'llama-2-ko-vllm': ('beomi/open-llama-2-ko-7b', None, LlamaTokenizer),
        'llama-2-ko-chat-vllm': ('kfkas/Llama-2-ko-7b-Chat', None, LlamaTokenizer),
        'llama-3-vllm': ('meta-llama/Meta-Llama-3-8B', None, AutoTokenizer),
        'llama-3-chat-vllm': ('meta-llama/Meta-Llama-3-8B-Instruct', None, AutoTokenizer),
        'llama-3-ko-vllm': ('beomi/Llama-3-Open-Ko-8B', None, AutoTokenizer),
        'llama-3-ko-chat-vllm': ('beomi/Llama-3-Open-Ko-8B-Instruct-preview', None, AutoTokenizer),
        'llama-2-ko-prime-base-en-qlora-bf16-vllm': (os.path.join(SCRIPT_DIR, '/data/sehyeong/nmt/models/onnx/llama2-prime-base'), None, LlamaTokenizer),
        'llama-3-ko-prime-base-en-qlora-bf16-vllm': ('traintogpb/llama-3-enko-translator-8b-qlora-bf16-upscaled', None, AutoTokenizer),
        'llama-3-prime-base-ja=qlora-bf16-vllm': ('/data/sehyeong/nmt/models/mmt_ft/en', None, AutoTokenizer),
        'llama-3-ko-prime-base-ja=qlora-bf16-vllm': ('/data/sehyeong/nmt/models/mmt_ft/ko', None, AutoTokenizer),
        'llama-3-ja-prime-base-ja=qlora-bf16-vllm': ('/data/sehyeong/nmt/models/mmt_ft/ja', None, AutoTokenizer),
        'llama-3-koja-scaled-halfja-prime-base-ja=qlora-bf16-vllm': ('/data/sehyeong/nmt/models/mmt_ft/koja-halfja', None, AutoTokenizer),
        'llama-3-kojazh-scaled-equal-prime-base-ja=qlora-bf16-vllm': ('/data/sehyeong/nmt/models/mmt_ft/kojazh-equal', None, AutoTokenizer),
        'llama-3-koja-scaled-sigmoid-prime-base-ja=qlora-bf16-vllm': ('/data/sehyeong/nmt/models/mmt_ft/koja-sigmoid', None, AutoTokenizer),
        'llama-3-koja-scaled-linear-prime-base-ja=qlora-bf16-vllm': ('/data/sehyeong/nmt/models/mmt_ft/koja-linear', None, AutoTokenizer),
        'llama-3-ko-prime-base-word-mix-all-bidir-qlora-bf16-vllm': (os.path.join(SCRIPT_DIR, '../../training/llama_qlora/models/llama3-word-mix-all-bidir-merged-bf16'), None, AutoTokenizer),
    }
    assert model_type in model_mapping.keys(), 'Wrong model type'

    model_name, model_cls, tokenizer_cls = model_mapping[model_type]
    if isinstance(model_name, tuple):
        model_name, adapter_path = model_name[0], model_name[1]
    
    if 'vllm' in model_type:
        model = LLM(model=model_name, seed=SEED, tensor_parallel_size=TENSOR_PARALLEL_SIZE)
    elif '16' in model_type:
        if model_type.endswith('bf16'):
            model = model_cls.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        elif model_type.endswith('fp16'):
            model = model_cls.from_pretrained(model_name, torch_dtype=torch.float16)
    elif 'qlora' in model_type:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        torch_dtype = torch.bfloat16
        model = model_cls.from_pretrained(
            model_name, 
            max_length=768 if model_type.startswith('llama') else 512,
            quantization_config=bnb_config, 
            attn_implementation='flash_attention_2' if 'prime' in model_type or model_type.startswith('alma') else None,
            torch_dtype=torch_dtype,
        )
        model = PeftModel.from_pretrained(
            model, 
            adapter_path, 
            torch_dtype=torch_dtype,
            adapter_name=model_type.split('-')[-1] if 'dpo' in model_type else 'default',
        )
    else:
        model = model_cls.from_pretrained(model_name)

    tokenizer = tokenizer_cls.from_pretrained(model_name)
    if model_type.startswith('llama'):
        if '3' in model_type:
            tokenizer.pad_token_id = 128002
        else:
            tokenizer.pad_token = "</s>"
            tokenizer.pad_token_id = 2
            tokenizer.eos_token = "<|endoftext|>"
            tokenizer.eos_token_id = 46332
        tokenizer.model_max_length = 768
        tokenizer.add_eos_token = True
    elif model_type.startswith('alma'):
        tokenizer.eos_token = "</s>"
        tokenizer.eos_token_id = 2
        tokenizer.add_eos_token = True
        tokenizer.model_max_length = 512

    return model, tokenizer


@torch.no_grad()
def translate(model, tokenizer, text, model_type, src_lang=None, tgt_lang=None, print_result=False, max_length=512):
    """
    Translate the input text using the pre-trained language model.

    Parameters:
    - model (PreTrainedModel): Pre-trained language model.
    - tokenizer (PreTrainedTokenizer): Pre-trained tokenizer.
    - text (str): Input text to be translated.
    - model_type (str): Type of the pre-trained language model.
    - print_result (bool): Whether to print the translated text.

    Returns:
    - translated_text (str): Translated text.
    """
    if not model_type.endswith('vllm'):
        model.to(DEVICE)

    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    if model_type.startswith('madlad'):
        input_text = f"<2{tgt_lang}> {text}"
    elif 'llama' in model_type:
        # if 'prime' in model_type:
        input_text = f"Translate this from {LANG_TABLE[src_lang]} to {LANG_TABLE[tgt_lang]}.\n### {LANG_TABLE[src_lang]}: {text}\n### {LANG_TABLE[tgt_lang]}:"
        if 'llama-3' in model_type and 'qlora' in model_type:
            input_text += " "
        # else:
            # input_text = f"### {LANG_TABLE[src_lang]}: {text}\n### {LANG_TABLE[tgt_lang]}: "
    elif model_type.startswith('mt5'):
        input_text = f"<{src_lang}> {text} <{tgt_lang}>"
    elif model_type.startswith('alma'):
        input_text = f"Translate this from {LANG_TABLE[src_lang]} to {LANG_TABLE[tgt_lang]}:\n{LANG_TABLE[src_lang]}: {text}\n{LANG_TABLE[tgt_lang]}:"

    if 'mbart' in model_type:
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang
    elif 'nllb' in model_type:
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang
    
    inputs = tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True)
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    if model_type.endswith('vllm'):
        use_beam_search = False
        if use_beam_search:
            # best_of만 custom 가능, 나머지는 고정
            best_of = 4
            temperature = 0.0
            top_k = -1
            top_p = 1.00
        else:
            # best_of는 무조건 1이고, 나머지는 custom 가능
            best_of = 1
            temperature = 0.0
            top_k = 40
            top_p = 0.95

        sampling_params = SamplingParams(
            temperature=temperature,
            use_beam_search=use_beam_search,
            best_of=best_of,
            top_k=top_k,
            top_p=top_p,
            skip_special_tokens=True,
            stop='<|endoftext|>',
            repetition_penalty=1.1,
            max_tokens=350
        )
        outputs = model.generate([input_text], sampling_params, use_tqdm=False)
        translated_text = outputs[0].outputs[0].text
    else:
        if (model_type.startswith('llama') and 'qlora' in model_type) or model_type.startswith('alma'):
            eos_token_id = 128001 if '3' in model_type else 46332
            if inputs['input_ids'][0][-1] == eos_token_id:
                inputs['input_ids'] = inputs['input_ids'][0][:-1].unsqueeze(dim=0)
                inputs['attention_mask'] = inputs['attention_mask'][0][:-1].unsqueeze(dim=0)
            outputs = model.generate(**inputs, max_length=max_length, eos_token_id=eos_token_id)
        elif model_type.startswith('madlad'):
            inputs['input_ids'] = inputs['input_ids'][0][1:].unsqueeze(dim=0)
            inputs['attention_mask'] = inputs['attention_mask'][0][1:].unsqueeze(dim=0)
            outputs = model.generate(**inputs, max_length=max_length)
        elif 'mbart' in model_type or 'nllb' in model_type:
            outputs = model.generate(**inputs, max_length=max_length, forced_bos_token_id=tokenizer.lang_code_to_id[tokenizer.tgt_lang])
        else:
            outputs = model.generate(**inputs, max_length=max_length)
        
        input_len = len(inputs['input_ids'].squeeze()) if model_type.startswith('llama') or model_type.startswith('alma') else 0
        translated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    translated_text = re.sub(r'\s+', ' ', translated_text)
    translated_text = translated_text.strip()
    
    if print_result:
        print(f"[Input] {text}")
        print(f"[Output] {translated_text}")

    return translated_text


def inference(
        model_type, 
        model, 
        tokenizer, 
        source_column, 
        target_column, 
        file_path, 
        src_lang=None, 
        tgt_lang=None, 
        print_result=False
    ):
    """
    Perform inference on the specified dataset using the specified pre-trained language model.

    Parameters:
    - model_type (str): Type of the pre-trained language model.
    - source_column (str): Name of the source column in the dataset.
    - target_column (str): Name of the target column in the dataset.
    - file_path (str): Path to the dataset file.
    - print_result (bool): Whether to print the translated text.
    """
    set_seed(SEED)
    if not model_type.endswith('vllm'):
        model.to(DEVICE)

    if 'llama' in model_type:
        max_length = 768
    elif 'madlad' in model_type or 'mt5' in model_type:
        max_length = 384
    else:
        max_length = 512

    eval_df = pd.read_csv(file_path)
    tqdm_iterator = tqdm(eval_df.iterrows(), total=len(eval_df), desc="Translating")
    translations, elapsed_times = [], []
    for _, row in tqdm_iterator:
        if 'direction' in eval_df.columns:
            src_lang, tgt_lang = row['direction'].split('-')
        else:
            src_lang = 'en' if src_lang is None else src_lang
            tgt_lang = 'ko' if tgt_lang is None else tgt_lang
        text = str(row[source_column])

        start_time = datetime.now()
        translation = translate(model, tokenizer, text, model_type, src_lang, tgt_lang, print_result, max_length)
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds() * 1000
        if print_result:
            print(f"Elapsed time: {elapsed_time:.1f} ms")

        translations.append(translation)
        elapsed_times.append(elapsed_time)

    eval_df[target_column] = translations
    target_time_column = target_column.replace('_trans', '_time')
    eval_df[target_time_column] = elapsed_times
    eval_df.to_csv(file_path, index=False)


def inference_single(
        model_type, 
        model, 
        tokenizer, 
        text, 
        src_lang=None, 
        tgt_lang=None
    ):
    """
    Perform inference on a single sentence using the specified pre-trained language model.

    Parameters:
    - model_type (str): Type of the pre-trained language model.
    - text (str): Input text to be translated.

    Returns:
    - translation (str): Translated text.
    """
    set_seed(SEED)
    if not model_type.endswith('vllm'):
        model.to(DEVICE)
    src_lang = 'en' if src_lang is None else src_lang
    tgt_lang = 'ko' if tgt_lang is None else tgt_lang
    start_time = datetime.now()
    translation = translate(model, tokenizer, text, model_type, src_lang, tgt_lang)
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds() * 1000
    print(f"Elapsed time: {elapsed_time:.2f} ms")

    return translation


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='llama-bf16', help="Pre-trained language model type for inference (e.g., llama-bf16)")
    parser.add_argument("--inference_type", type=str, default='sentence', help="Inference type (sentence or dataset)")
    parser.add_argument("--dataset", type=str, default='sample', help="Dataset path for inference (only for dataset inference, preset: aihub / flores / sample, or custom path to a CSV file)")
    parser.add_argument("--sentence", type=str, default="NMIXX is a South Korean girl group that made a comeback on January 15, 2024 with their new song 'DASH'.", help="Input English text for inference (only for sentence inference)")
    parser.add_argument("--src_lang", type=str, default="en", help="Source language code (just for sentence translation; default: en)")
    parser.add_argument("--tgt_lang", type=str, default="ko", help="Target language code (just for sentence translation; default: ko)")
    parser.add_argument("--print_result", action="store_true", help="Print the translated text")
    args = parser.parse_args()
    dataset = args.dataset
    
    if args.dataset.endswith('.csv'):
        file_path = args.dataset
    else:
        file_path_dict = {
            'sample': os.path.join(SCRIPT_DIR, "../results/sample.csv"),
            'aihub': os.path.join(SCRIPT_DIR, "../results/test_tiny_uniform100_inferenced.csv"),
            'flores': os.path.join(SCRIPT_DIR, "../results/test_flores_inferenced.csv"),
            'prime-api': os.path.join(SCRIPT_DIR, "../results/prime/test_sparta_bidir_api_inferenced.csv"),
            'prime-llama2': os.path.join(SCRIPT_DIR, "../results/prime/test_sparta_bidir_llama2_inferenced.csv"),
            'prime-llama3': os.path.join(SCRIPT_DIR, "../results/prime/test_sparta_bidir_llama3_inferenced.csv"),
            'dpo': os.path.join(SCRIPT_DIR, "../results/koen_dpo_bidir_inferenced.csv"),
            'ko_words': os.path.join(SCRIPT_DIR, "../results/words/ko_word_test_1k.csv"),
            'en_words': os.path.join(SCRIPT_DIR, "../results/words/en_word_test_1k.csv"),
        }
        file_path = file_path_dict[dataset]

    model_type_dict = {
        'llama-2': 'llama-2-vllm',
        'llama-2-chat': 'llama-2-chat-vllm',
        'llama-2-ko': 'llama-2-ko-vllm',
        'llama-2-ko-chat': 'llama-2-ko-chat-vllm',
        'llama-3': 'llama-3-vllm',
        'llama-3-chat': 'llama-3-chat-vllm',
        'llama-3-ko': 'llama-3-ko-vllm',
        'llama-3-ko-chat': 'llama-3-ko-chat-vllm',
        'llama-2-ko-prime-base-en': 'llama-2-ko-prime-base-en-qlora-bf16-vllm',
        'llama-3-ko-prime-base-en': 'llama-3-ko-prime-base-en-qlora-bf16-vllm',
        'llama-3-prime-base-ja': 'llama-3-ko-prime-base-ja=qlora-bf16-vllm',
        'llama-3-ko-prime-base-ja': 'llama-3-ko-prime-tiny-ja-qlora-bf16-vllm',
        'llama-3-ja-prime-base-ja': 'llama-3-ja-prime-tiny-ja-qlora-bf16-vllm',
        'llama-3-koja-halfja-prime-base-ja': 'llama-3-koja-scaled-halfja-prime-tiny-ja-qlora-bf16-vllm',
        'llama-3-kojazh-equal-prime-base-ja': 'llama-3-kojazh-scaled-equal-prime-tiny-ja-qlora-bf16-vllm',
        'llama-3-koja-sigmoid-prime-base-ja': 'llama-3-koja-scaled-sigmoid-prime-tiny-ja-qlora-bf16-vllm',
        'llama-3-koja-linear-prime-base-ja': 'llama-3-koja-scaled-linear-prime-tiny-ja-qlora-bf16-vllm',
        'llama-3-ko-mix-all-bidir': 'llama-3-ko-prime-base-en-word-mix-all-bidir-qlora-bf16-vllm',
    }
    
    model_type = model_type_dict[args.model_type]
    model, tokenizer = load_model_and_tokenizer(model_type)

    print(f"Inference model: {model_type.upper()}")
    print(f"Inference type: {args.inference_type.upper()}")
    if args.inference_type == 'dataset':
        print(f"Dataset: {dataset.upper()}")
    elif args.inference_type == 'sentence':
        print(f"Sentence: {args.sentence}")

    if args.inference_type == 'dataset':
        source_column = "src" if any(args.dataset.startswith(bidir_data) for bidir_data in ['prime', 'sample', 'dpo']) else args.src_lang
        target_column = model_type + '_trans'
        inference(
            model_type, 
            model, 
            tokenizer, 
            source_column, 
            target_column, 
            file_path, 
            args.src_lang,
            args.tgt_lang,
            print_result=args.print_result,
        )
    
    if args.inference_type == 'sentence':
        translation = inference_single(
            model_type, 
            model,
            tokenizer,
            args.sentence, 
            args.src_lang, 
            args.tgt_lang,
        )
        print(f"Translation: {translation}")
