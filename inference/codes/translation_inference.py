import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSOR_PARALLEL_SIZE = 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_USE_MODELSCOPE'] = 'false'
import re
import sys
import yaml
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import pandas as pd

sys.path.append('./')
sys.path.append(os.path.join(SCRIPT_DIR, '../../'))
from training.training_utils import set_seed
from inference.codes.translation_info import *
from translators import ApiTranslator, HfTranslator, VllmTranslator
from translators import make_prompt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)


def load_translator(args):
    
    set_seed(args.seed)

    if args.model_type == 'api':
        model_path = args.model_name
    else:
        if args.lora_name is None:
            model_path = MODEL_MAPPING[args.model_name]
            lora_path = None
        else:
            model_path = PLM_MAPPING[args.model_name]
            lora_path = ADAPTER_MAPPING[args.lora_name]

    if 'hf' in args.model_type:
        translator = HfTranslator(model_path, max_length=args.max_length)
        translator.model = translator.load_model(
            lora_path=lora_path,
            lora_nickname=args.lora_nickname,
            quantization='nf4' if args.model_type == 'hf-qlora' else None,
            torch_dtype=args.torch_dtype,
            cache_dir=HF_CACHE_DIR
        )
        translator.tokenizer = translator.load_tokenizer(padding_side='left')
    
    elif args.model_type == 'vllm':
        translator = VllmTranslator(model_path, lora_path=lora_path, lora_nickname=args.lora_nickname)
        translator.model = translator.load_model(
            max_length=MAX_LENGTH,
            max_lora_rank=VLLM_MAX_LORA_RANK,
            seed=SEED,
            torch_dtype=torch.bfloat16,
            cache_dir=HF_CACHE_DIR,
            vram_limit=VLLM_VRAM_LIMIT,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE
        )
        translator.sampling_params = translator.load_sampling_params(
            temperature=0.0, 
            use_beam_search=False, 
            best_of=1,
            top_k=40, 
            top_p=0.95, 
            skip_special_tokens=False, 
            stop=None, 
            repetition_penalty=1.1, 
            max_tokens=MAX_LENGTH
        )
    
    elif args.model_type == 'api':
        translator = ApiTranslator(model_path)
        translator.model = translator.load_model()
    
    return translator


def translate_with_time(trans_func, prompts, *args, **kwargs):
    start_time = datetime.now()
    translations = trans_func(prompts, *args, **kwargs)
    end_time = datetime.now()
    translation_time_in_ms = (end_time - start_time).total_seconds() * 1000
    translation_time_in_ms = round(translation_time_in_ms, 3)
    return translations, translation_time_in_ms


def preprocess_text(text):
    text = str(text)
    text = text.strip()
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text


def postprocess_text(text, capitalize=True):
    text = str(text)
    if re.search(r'(</\w+>\s*){3}$', text):
        text = re.sub(r'(</\w+>\s*){1,3}$', '', text)
    text = text.strip()
    if capitalize:
        text = text.capitalize()
    return text


def model_translate(translator, texts, src_lang, tgt_lang, args):
    texts = [preprocess_text(text) for text in texts]

    trans_func = translator.translate

    if isinstance(src_lang, str):
        src_lang = [src_lang] * len(texts)
    if isinstance(tgt_lang, str):
        tgt_lang = [tgt_lang] * len(texts)
    
    if args.model_type == 'api':
        prompts = texts
    else:
        prompts = [make_prompt(text, src, tgt, args.guidelines, args.prompt_type) for text, src, tgt in zip(texts, src_lang, tgt_lang)]
    
    batch_prompts = [prompts[i:i+args.batch_size] for i in range(0, len(prompts), args.batch_size)]
    batch_texts = [texts[i:i+args.batch_size] for i in range(0, len(texts), args.batch_size)]

    translations, translation_times = [], []
    try:
        for batch_prompt, batch_text in tqdm(zip(batch_prompts, batch_texts), total=len(batch_prompts), desc="Translating..."):
            if args.model_type == 'api':
                trans_args = (src_lang, tgt_lang)
            else:
                trans_args = ()
            translations_tmp, translation_time_tmp = translate_with_time(trans_func, batch_prompt, *trans_args)
            translation_times_tmp = [translation_time_tmp / len(batch_prompt)] * len(batch_prompt)
            if args.print_result:
                for tgt, trans_tmp in zip(batch_text, translations_tmp):
                    print(f"\n[INPUT] {tgt}")
                    print(f"[OUTPUT] {postprocess_text(trans_tmp)}")
                    print(f"[AVG TIME] {translation_time_tmp / len(batch_prompt):.3f} ms (for {len(batch_prompt)} samples)")
            translations.extend(translations_tmp)
            translation_times.extend(translation_times_tmp)
    except Exception as e:
        print(f"An error occurred: {e}")
        remaining_length = len(texts) - len(translations)
        translations.extend([None] * remaining_length)
        translation_times.extend([None] * remaining_length)
    
    translations = [postprocess_text(trans_tmp, capitalize=False) if trans_tmp is not None else None for trans_tmp in translations]
    
    return translations, translation_times


def translate_text(translator, text, args):
    texts = [text]
    translation, translation_time = model_translate(translator, texts, args)
    return translation[0], translation_time[0]


def translate_df(translator, df, args):
    texts = df[args.tgt_col].tolist()
    
    if args.lang_col is None and (args.src_lang is None and args.tgt_lang is None):
        raise ValueError("lang_col or (src_lang, tgt_lang) pair must be provided.")
    if args.lang_col is None:
        src_lang, tgt_lang = args.src_lang, args.tgt_lang
    else:
        src_lang, tgt_lang = [], []
        for lang_pair in df[args.lang_col]:
            src, tgt = lang_pair.split('-')
            src_lang.append(src)
            tgt_lang.append(tgt)

    resume_idx = 0
    if (args.trans_col in df.columns) and (not pd.isna(df[args.trans_col][0])):
        resume_idx = df[args.trans_col].isna().idxmax()
        texts = texts[resume_idx:]
        if isinstance(src_lang, list):
            src_lang = src_lang[resume_idx:]
        if isinstance(tgt_lang, list):
            tgt_lang = tgt_lang[resume_idx:]
        print(f"Resuming from index {resume_idx}...")

    translations, translation_times = model_translate(translator, texts, src_lang, tgt_lang, args)
    
    if resume_idx > 0:
        translations = df[args.trans_col].tolist()[:resume_idx] + translations
        translation_times = df[args.time_col].tolist()[:resume_idx] + translation_times

    if args.trans_col not in df.columns:
        df.insert(df.columns.get_loc(args.tgt_col)+1, args.trans_col, translations)
    else:
        df[args.trans_col] = translations
    if len(translation_times) > 0 and args.time_col is not None:
        if args.time_col not in df.columns:
            df.insert(df.columns.get_loc(args.tgt_col)+2, args.time_col, translation_times)
        else:
            df[args.time_col] = translation_times

    return df


def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parse_infer_args(config_path):
    config = load_yaml_config(config_path)
    model_config = config['model']
    data_config = config['data']
    infer_config = config['infer']

    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model_name', 
                        type=str, 
                        default=model_config['model_name'], 
                        help="Translation model name (prefixed in translation_info.py) to use.")
    parser.add_argument('--lora_name', 
                        type=lambda x: None if x.lower() == 'none' else x, 
                        default=model_config['lora_name'], 
                        help="LoRA adapter's name (prefixed in translation_info.py) to use.")
    parser.add_argument('--lora_nickname',
                        type=lambda x: None if x.lower() == 'none' else x,
                        default=model_config['lora_nickname'],
                        help="LoRA adapter's nickname to use.")
    parser.add_argument('--model_type', 
                        type=str, 
                        default=model_config['model_type'], 
                        help="Inference type: (api, hf, hf-qlora, vllm)")
    parser.add_argument('--torch_dtype',
                        type=lambda x: getattr(torch, x) if hasattr(torch, x) else None,
                        default=model_config['torch_dtype'],
                        help="Torch dtype to use.")
    parser.add_argument('--max_length',
                        type=int,
                        default=model_config['max_length'],
                        help="Max length for translation.")
    parser.add_argument('--seed',
                        type=int,
                        default=model_config['seed'],
                        help="Random seed for translation.")
    # Data
    parser.add_argument('--prompt_type', 
                        type=lambda x: None if x.lower() == 'none' else x,
                        default=data_config['prompt_type'], 
                        help="Translation prompt type to use.")
    parser.add_argument('--guidelines', 
                        type=lambda x: x.split(','), 
                        default=data_config['guidelines'], 
                        help="Guidelines to use for translation.")
    parser.add_argument('--data_type', 
                        type=str, 
                        default=data_config['data_type'], 
                        help="Translate a dataset or a sentence.")
    parser.add_argument('--sentence_text', 
                        type=str, 
                        default=data_config['sentence']['text'], 
                        help="Sentence to translate.")
    parser.add_argument('--dataset_name', 
                        type=lambda x: None if x.lower() == 'none' else x,
                        default=data_config['dataset']['dataset_name'], 
                        help="Dataset to translate.")
    parser.add_argument('--tgt_col', 
                        type=lambda x: None if x.lower() == 'none' else x,
                        default=data_config['dataset']['tgt_col'], 
                        help="Target column to translate.")
    parser.add_argument('--lang_col', 
                        type=lambda x: None if x.lower() == 'none' else x, 
                        default=data_config['dataset']['lang_col'], 
                        help="Language direction column to translate.")
    parser.add_argument('--trans_col', 
                        type=lambda x: None if x.lower() == 'none' else x, 
                        default=data_config['dataset']['trans_col'], 
                        help="Translated column to save.")
    parser.add_argument('--time_col', 
                        type=lambda x: None if x.lower() == 'none' else x, 
                        default=data_config['dataset']['time_col'], 
                        help="Translation time column to save.")
    # Inference
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=infer_config['batch_size'], 
                        help="Batch size for translation.")
    parser.add_argument('--src_lang', 
                        type=lambda x: None if x.lower() == 'none' else x, 
                        default=infer_config['src_lang'], 
                        help="Source language.")
    parser.add_argument('--tgt_lang', 
                        type=lambda x: None if x.lower() == 'none' else x, 
                        default=infer_config['tgt_lang'], 
                        help="Target language.")
    parser.add_argument('--print_result', 
                        type=lambda x: (str(x).lower() == 'true'), 
                        default=infer_config['print_result'],
                        help="Print the translation result in dataset inference.")

    args = parser.parse_args()

    if args.sentence_text is None and args.dataset_name is None:
        raise ValueError("Either sentence or dataset should be provided.")

    return args


def print_infer_info(args):
    print("\n############### TRANLSATION INFO ###############")
    print(f"MODEL: {args.model_name.upper()}")
    if args.lora_name is not None:
        print(f"ADAPTER: {args.lora_name.upper()}")
    print(f"INFERENCE: {args.model_type.upper()}")
    if args.data_type == 'sentence':
        print(f"SENTENCE: {args.sentence_text}")
    else:
        print(f"DATASET: {args.dataset_name.upper()}")
    print(f"SOURCE LANG: {str(args.src_lang).upper()}")
    print(f"TARGET LANG: {str(args.tgt_lang).upper()}")
    print("################################################\n")


def main():
    yaml_path = os.path.join(SCRIPT_DIR, 'translation_config.yaml')
    args = parse_infer_args(yaml_path)
    print_infer_info(args)

    translator = load_translator(args)

    if args.data_type == 'sentence':
        text = args.sentence_text
        translation, translation_time = translate_text(translator, text, args)
        print("\n############### TRANSLATION RESULT ###############")
        print(f"TRANSLATION: {translation}")
        print(f"TRANSLATION TIME: {translation_time} ms")
        print("#################################################\n")

    elif args.data_type == 'dataset':
        if args.dataset_name.endswith('.csv'):
            df_path = args.dataset_name
        else:
            df_path = DF_PATH_DICT[args.dataset_name]

        df = pd.read_csv(df_path)
        df = translate_df(translator, df, args)
        df.to_csv(df_path, index=False)

    else:
        raise ValueError("data_type should be either 'sentence' or 'dataset'.")
    

if __name__ == '__main__':
    main()
