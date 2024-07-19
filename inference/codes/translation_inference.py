import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSOR_PARALLEL_SIZE = 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_USE_MODELSCOPE'] = 'false'
import sys
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import pandas as pd
from peft import PeftModel
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from deepl import Translator as DeeplTranslator
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig

sys.path.append('./')
sys.path.append(os.path.join(SCRIPT_DIR, '../../'))
from api_translator import PapagoTranslator
from api_secret import (
    PAPAGO_CLIENT_ID,
    PAPAGO_CLIENT_SECRET,
)
from api_secret import (
    DEEPL_CLIENT_KEY,
)
from training.training_utils import set_seed
from inference_info import *


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)
        

def papago_translate(text, src_lang='ko', tgt_lang='en'):
    translator = PapagoTranslator(PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET)
    translation = translator.translate(text, 
                                       src_lang=PAPAGO_LANG_CODE[src_lang], 
                                       tgt_lang=PAPAGO_LANG_CODE[tgt_lang])
    return translation


def deepl_translate(text, src_lang='ko', tgt_lang='en'):
    translator = DeeplTranslator(DEEPL_CLIENT_KEY)
    translation = translator.translate(text, 
                                       src_lang=DEEPL_LANG_CODE[src_lang]['src'], 
                                       tgt_lang=DEEPL_LANG_CODE[tgt_lang]['tgt'])
    return translation


def load_hf_model(model_path, 
                  lora_path=None, 
                  adapter_name=None, 
                  max_length=MAX_LENGTH,
                  quantization_config=None,
                  torch_dtype=torch.bfloat16):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            max_length=max_length,
            quantization_config=quantization_config,
            attn_implementation='flash_attention_2',
            torch_dtype=torch_dtype, 
            cache_dir=HF_CACHE_DIR
        )
    except:
        print(f"Model {model_path} is not a CaualLM, trying Seq2SeqLM...")
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path, 
                max_length=max_length,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype, 
                cache_dir=HF_CACHE_DIR
            )
        except:
            raise ValueError(f"Model {model_path} is not a CausalLM or Seq2SeqLM neither. Try again with a proper model.")
        
    model.to(DEVICE)

    if lora_path is not None:
        model = PeftModel.from_pretrained(
            model,
            lora_path,
            adapter_name=adapter_name,
            torch_dtype=torch_dtype,
        )

    return model


def load_hf_tokenizer(model_type, src_lang, tgt_lang, padding_side='right'):
    model_path = MODEL_MAPPING[model_type]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = padding_side
    tokenizer.model_max_length = MAX_LENGTH
    if 'llama' in model_type:
        if any(model_suffix in model_type.lower() for model_suffix in ['llama-2', 'llama2']):
            tokenizer.pad_token_id = 2
            tokenizer.eos_token_id = 46332
        elif any(model_suffix in model_type.lower() for model_suffix in ['llama-3', 'llama3']):
            tokenizer.pad_token_id = 128002
    elif 'mbart' in model_type:
        tokenizer.src_lang = MBART_LANG_CODE[src_lang]
        tokenizer.tgt_lang = MBART_LANG_CODE[tgt_lang]
    try:
        tokenizer.add_eos_token = True
    except:
        print(f"Tokenizer {model_path} does not support 'add_eos_token'.")
    
    return tokenizer


def load_vllm_model(model_path, 
                    lora_path=None,
                    torch_dtype=torch.bfloat16,
                    seed=SEED, 
                    max_length=MAX_LENGTH,
                    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
                    vram_limit=VLLM_VRAM_LIMIT,
                    max_lora_rank=VLLM_MAX_LORA_RANK):
    model = LLM(
        model=model_path,
        tokenizer=model_path,
        max_seq_len_to_capture=max_length,
        enable_lora=True if lora_path is not None else False,
        max_lora_rank=max_lora_rank if lora_path is not None else None,
        seed=seed,
        dtype=torch_dtype,
        trust_remote_code=True,
        download_dir=HF_CACHE_DIR,
        gpu_memory_utilization=vram_limit,
        tensor_parallel_size=tensor_parallel_size,
    )
    return model


def load_vllm_params(temperature=0.0,
                     use_beam_search=False,
                     top_k=40,
                     top_p=0.95,
                     skip_special_tokens=True,
                     stop=None,
                     repetition_penalty=1.1,
                     max_tokens=MAX_LENGTH):
    sampling_params = SamplingParams(
        temperature=temperature,
        use_beam_search=use_beam_search,
        top_k=top_k,
        top_p=top_p,
        skip_special_tokens=skip_special_tokens,
        stop=stop,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
    )
    return sampling_params


def load_model(model_type, 
               inference_type, 
               lora_type=None, 
               adapter_name=None,
               max_length=MAX_LENGTH,
               torch_dtype=torch.bfloat16,
               seed=SEED,
               tensor_parallel_size=TENSOR_PARALLEL_SIZE,
               vram_limit=VLLM_VRAM_LIMIT,
               max_lora_rank=VLLM_MAX_LORA_RANK):
    
    set_seed(seed)

    if lora_type is None:
        model_path = MODEL_MAPPING[model_type]
        lora_path = None
    else:
        model_path = PLM_MAPPING[model_type]
        lora_path = ADAPTER_MAPPING[lora_type]

    if inference_type in ['hf', 'qlora']:
        if inference_type == 'qlora':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )

        model = load_hf_model(
            model_path,
            lora_path=lora_path if lora_type is not None else None,
            adapter_name=adapter_name,
            max_length=max_length,
            quantization_config=bnb_config if inference_type == 'qlora' else None,
            torch_dtype=torch_dtype
        )
    
    elif inference_type == 'vllm':
        model = load_vllm_model(
            model_path,
            lora_path=lora_path,
            seed=seed,
            max_length=max_length,
            tensor_parallel_size=tensor_parallel_size,
            vram_limit=vram_limit,
            max_lora_rank=max_lora_rank,
        )
    
    return model


def make_prompt(text, src_lang, tgt_lang, prompt_type=None):
    if tgt_lang is None:
        raise ValueError("tgt_lang must be provided.")
    
    if prompt_type is None:
        prompt = text
    elif prompt_type == 'llama':
        if src_lang is None:
            raise ValueError("src_lang must be provided.")
        instruction = f"Translate this from {LLAMA_LANG_TABLE[src_lang]} to {LLAMA_LANG_TABLE[tgt_lang]}."
        src_suffix = f"### {LLAMA_LANG_TABLE[src_lang]}:"
        tgt_suffix = f"### {LLAMA_LANG_TABLE[tgt_lang]}:"
        prompt = f"{instruction}\n{src_suffix} {text}\n{tgt_suffix}"
    elif prompt_type == 'madlad':
        src_suffix = f"<2{MADLAD_LANG_CODE[tgt_lang]}>"
        prompt = f"{src_suffix} {text}"
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")
    
    return prompt


def hf_translate(prompts, model, tokenizer):
    inputs = tokenizer(
        prompts, 
        return_tensors='pt', 
        padding=True, 
        truncation=True,
        max_length=MAX_LENGTH
    )
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    for idx, (input_ids, attn_mask) in enumerate(zip(inputs['input_ids'], inputs['attention_mask'])):
        if input_ids[-1] != tokenizer.eos_token_id:
            continue
        inputs['input_ids'][idx] = input_ids[:-1]
        inputs['attention_mask'][idx] = attn_mask[:-1]

    outputs = model.generate(
        **inputs, 
        max_length=tokenizer.model_max_length, 
        eos_token_id=tokenizer.eos_token_id
    )
    outputs = [outputs[idx][inputs['input_ids'][idx].shape[0]:] for idx in range(len(outputs))]
    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    translations = [translation.strip() for translation in translations]
    
    return translations


def vllm_translate(prompts, model, sampling_params, lora_path=None, adapter_name=None):
    lora_request = LoRARequest(adapter_name, 1, lora_path) if lora_path is not None else None
    outputs = model.generate(prompts, sampling_params, lora_request=lora_request, use_tqdm=False)
    translations = [output.outputs[0].text.strip() for output in outputs]
    return translations


def translate_with_time(trans_func, *args, **kwargs):
    start_time = datetime.now()
    translations = trans_func(*args, **kwargs)
    end_time = datetime.now()
    translation_time_in_ms = (end_time - start_time).total_seconds() * 1000
    translation_time_in_ms = round(translation_time_in_ms, 3)
    return translations, translation_time_in_ms


def model_translate(model,
                    texts,
                    tokenizer=None,
                    sampling_params=None,
                    batch_size=16,
                    lora_type=None,
                    inference_type='hf',
                    prompt_type=None,
                    src_lang=None,
                    tgt_lang=None,
                    print_result=False):

    if inference_type != 'vllm':
        trans_func = hf_translate
        trans_args = (texts, model, tokenizer)
    elif inference_type == 'vllm':
        trans_func = vllm_translate
        trans_args = (texts, model, sampling_params, lora_type)
    else:
        raise ValueError(f"Invalid inference type: {inference_type}")

    lora_path = ADAPTER_MAPPING[lora_type] if lora_type is not None else None

    if isinstance(src_lang, str):
        src_lang = [src_lang] * len(texts)
    if isinstance(tgt_lang, str):
        tgt_lang = [tgt_lang] * len(texts)
    prompts = [make_prompt(text, src, tgt, prompt_type) for text, src, tgt in zip(texts, src_lang, tgt_lang)]

    batch_prompts = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]

    translations, translation_times = [], []
    for batch_idx, batch in enumerate(tqdm(batch_prompts, desc="Translating bathces...")):
        if inference_type != 'vllm':
            trans_func = hf_translate
            trans_args = (batch, model, tokenizer)
        elif inference_type == 'vllm':
            trans_func = vllm_translate
            trans_args = (batch, model, sampling_params, lora_path)
        translations_tmp, translation_time_tmp = translate_with_time(trans_func, *trans_args)
        translation_times_tmp = [translation_time_tmp / len(batch)] * len(batch)
        if print_result:
            batch_texts = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
            for tgt, trans_tmp in zip(batch_texts[batch_idx], translations_tmp):
                print(f"\n[INPUT] {tgt}")
                print(f"[OUTPUT] {trans_tmp}")
                print(f"[AVG TIME] {translation_time_tmp / len(batch):.3f} ms (for {len(batch)} samples)")
        translations.extend(translations_tmp)
        translation_times.extend(translation_times_tmp)
    
    return translations, translation_times


def translate_text(model, 
                   text, 
                   tokenizer=None, 
                   sampling_params=None, 
                   lora_type=None, 
                   inference_type='hf', 
                   prompt_type=None, 
                   src_lang=None, 
                   tgt_lang=None):
    
    texts = [text]
    translation, translation_time = model_translate(
        model, 
        texts, 
        tokenizer=tokenizer, 
        sampling_params=sampling_params, 
        lora_type=lora_type, 
        inference_type=inference_type, 
        prompt_type=prompt_type, 
        src_lang=src_lang, 
        tgt_lang=tgt_lang
    )
    return translation[0], translation_time[0]


def translate_df(model,
                 df,
                 tgt_col,
                 lang_col=None,
                 trans_col='trans',
                 time_col='trans_time',
                 batch_size=16,
                 lora_type=None,
                 tokenizer=None,
                 sampling_params=None,
                 inference_type='hf',
                 prompt_type=None,
                 src_lang=None,
                 tgt_lang=None,
                 print_result=False):
    
    texts = df[tgt_col].tolist()

    if lang_col is None and (src_lang is None and tgt_lang is None):
        raise ValueError("lang_col or (src_lang, tgt_lang) pair must be provided.")
    if lang_col is not None:
        lang_pairs = df[lang_col].str.split('-', expand=True)
        src_lang = lang_pairs[0].tolist()
        tgt_lang = lang_pairs[1].tolist()

    translations, translation_times = model_translate(
        model, 
        texts, 
        tokenizer=tokenizer, 
        sampling_params=sampling_params, 
        batch_size=batch_size,
        lora_type=lora_type, 
        inference_type=inference_type, 
        prompt_type=prompt_type, 
        src_lang=src_lang, 
        tgt_lang=tgt_lang,
        print_result=print_result
    )
    
    df.insert(df.columns.get_loc(tgt_col)+1, trans_col, translations)
    if len(translation_times) > 0:
        df.insert(df.columns.get_loc(tgt_col)+2, time_col, translation_times)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, help="Translation model type to use.")
    parser.add_argument('--lora_type', type=str, default=None, help="LoRA adapter type to use.")
    parser.add_argument('--prompt_type', type=str, default=None, help="Translation prompt type to use.")
    parser.add_argument('--inference_type', type=str, default='vllm', help="Inference type: (hf, vllm, qlora)")
    parser.add_argument('--data_type', type=str, default='sentence', help="Translate a dataset or a sentence.")
    parser.add_argument('--sentence', type=str, default="안녕, 세상아", help="Sentence to translate.")
    parser.add_argument('--dataset', type=str, default=None, help="Dataset to translate.")
    parser.add_argument('--tgt_col', type=str, default=None, help="Target column to translate.")
    parser.add_argument('--lang_col', type=str, default=None, help="Language direction column to translate.")
    parser.add_argument('--trans_col', type=str, default=None, help="Translated column to save.")
    parser.add_argument('--time_col', type=str, default=None, help="Translation time column to save.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for translation.")
    parser.add_argument('--src_lang', type=str, default='ko', help="Source language.")
    parser.add_argument('--tgt_lang', type=str, default='en', help="Target language.")
    parser.add_argument('--print_result', action='store_true', help="Print the translation result in dataset inference.")
    args = parser.parse_args()
    
    if args.sentence is None and args.dataset is None:
        raise ValueError("Either sentence or dataset should be provided.")

    print("\n############### TRANLSATION INFO ###############")
    print(f"MODEL: {args.model_type.upper()}")
    if args.lora_type is not None:
        print(f"ADAPTER: {args.lora_type.upper()}")
    print(f"INFERENCE: {args.inference_type.upper()}")
    if args.dataset is None:
        print(f"SENTENCE: {args.sentence}")
    else:
        print(f"DATASET: {args.dataset.upper()}")
    print(f"SOURCE LANG: {args.src_lang.upper()}")
    print(f"TARGET LANG: {args.tgt_lang.upper()}")
    print("################################################\n")

    model = load_model(
        args.model_type,
        args.inference_type,
        lora_type=args.lora_type,
        adapter_name=args.lora_type,
        max_length=MAX_LENGTH,
        torch_dtype=torch.bfloat16,
        seed=SEED,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        vram_limit=VLLM_VRAM_LIMIT,
        max_lora_rank=VLLM_MAX_LORA_RANK
    )
    tokenizer = None
    sampling_params = None
    if args.inference_type != 'vllm':
        if 'llama' in args.model_type:
            padding_side = 'left'
        else:
            padding_side = 'right'
        tokenizer = load_hf_tokenizer(args.model_type, args.src_lang, args.tgt_lang, padding_side)
    elif args.inference_type == 'vllm':
        sampling_params = load_vllm_params(
            temperature=0.0,
            use_beam_search=False,
            top_k=40,
            top_p=0.95,
            skip_special_tokens=True,
            stop=None,
            repetition_penalty=1.1,
            max_tokens=MAX_LENGTH
        )

    if args.data_type == 'sentence':
        text = args.sentence
        translation, translation_time = translate_text(
            model,
            text,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            lora_type=args.lora_type,
            inference_type=args.inference_type,
            prompt_type=args.prompt_type,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang
        )
        print("\n############### TRANSLATION RESULT ###############")
        print(f"TRANSLATION: {translation}")
        print(f"TRANSLATION TIME: {translation_time} ms")
        print("#################################################\n")

    elif args.data_type == 'dataset':
        if args.dataset.endswith('.csv'):
            df_path = args.dataset
        else:
            df_path = DF_PATH_DICT[args.dataset]

        df = pd.read_csv(df_path)
        df = translate_df(
            model,
            df,
            args.tgt_col,
            args.lang_col,
            trans_col=args.trans_col,
            time_col=args.time_col,
            batch_size=args.batch_size,
            lora_type=args.lora_type,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            inference_type=args.inference_type,
            prompt_type=args.prompt_type,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            print_result=args.print_result
        )
        df.to_csv(df_path, index=False)

    else:
        raise ValueError("data_type should be either 'sentence' or 'dataset'.")
    

if __name__ == '__main__':
    main()