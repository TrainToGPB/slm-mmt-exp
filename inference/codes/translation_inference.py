import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSOR_PARALLEL_SIZE = 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_USE_MODELSCOPE'] = 'false'
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
        if args.lora_type is None:
            model_path = MODEL_MAPPING[args.model_name]
            lora_path = None
        else:
            model_path = PLM_MAPPING[args.model_name]
            lora_path = ADAPTER_MAPPING[args.lora_type]

    if 'hf' in args.model_type:
        translator = HfTranslator(model_path, max_length=args.max_length)
        translator.model = translator.load_model(
            lora_path=lora_path,
            adapter_name=args.adapter_name,
            quantization='nf4' if args.model_type == 'hf-qlora' else None,
            torch_dtype=args.torch_dtype,
            cache_dir=HF_CACHE_DIR
        )
        translator.tokenizer = translator.load_tokenizer(padding_side='left')
    
    elif args.model_type == 'vllm':
        translator = VllmTranslator(model_path, lora_path=lora_path, adapter_name=args.adapter_name)
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
            skip_special_tokens=True, 
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


def postprocess_text(text, capitalize=True):
    text = str(text)
    text = text.strip()
    if capitalize:
        text = text.capitalize()
    return text


def model_translate(translator,
                    texts,
                    batch_size=16,
                    model_type='hf',
                    prompt_type=None,
                    src_lang=None,
                    tgt_lang=None,
                    print_result=False):

    trans_func = translator.translate

    if isinstance(src_lang, str):
        src_lang = [src_lang] * len(texts)
    if isinstance(tgt_lang, str):
        tgt_lang = [tgt_lang] * len(texts)
    
    if model_type == 'api':
        prompts = texts
    else:
        prompts = [make_prompt(text, src, tgt, prompt_type) for text, src, tgt in zip(texts, src_lang, tgt_lang)]

    batch_prompts = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    batch_texts = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]

    translations, translation_times = [], []
    for batch_prompt, batch_text in tqdm(zip(batch_prompts, batch_texts), total=len(batch_prompts), desc="Translating..."):
        if model_type == 'api':
            trans_args = (src_lang, tgt_lang)
        else:
            trans_args = ()
        translations_tmp, translation_time_tmp = translate_with_time(trans_func, batch_prompt, *trans_args)
        translation_times_tmp = [translation_time_tmp / len(batch_prompt)] * len(batch_prompt)
        if print_result:
            for tgt, trans_tmp in zip(batch_text, translations_tmp):
                print(f"\n[INPUT] {tgt}")
                print(f"[OUTPUT] {trans_tmp}")
                print(f"[AVG TIME] {translation_time_tmp / len(batch_prompt):.3f} ms (for {len(batch_prompt)} samples)")
        translations.extend(translations_tmp)
        translation_times.extend(translation_times_tmp)

    translations = [postprocess_text(trans_tmp, capitalize=False) for trans_tmp in translations]
    
    return translations, translation_times


def translate_text(translator, text, args):
    
    texts = [text]
    translation, translation_time = model_translate(
        translator, 
        texts, 
        model_type=args.model_type, 
        prompt_type=args.prompt_type, 
        src_lang=args.src_lang, 
        tgt_lang=args.tgt_lang
    )
    return translation[0], translation_time[0]


def translate_df(translator, df, args):
    texts = df[args.tgt_col].tolist()

    if args.lang_col is None and (src_lang is None and tgt_lang is None):
        raise ValueError("lang_col or (src_lang, tgt_lang) pair must be provided.")
    if args.lang_col is not None:
        lang_pairs = df[args.lang_col].str.split('-', expand=True)
        src_lang = lang_pairs[0].tolist()
        tgt_lang = lang_pairs[1].tolist()

    translations, translation_times = model_translate(
        translator, 
        texts, 
        batch_size=args.batch_size,
        model_type=args.model_type, 
        prompt_type=args.prompt_type, 
        src_lang=src_lang, 
        tgt_lang=tgt_lang,
        print_result=args.print_result
    )
    
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
                        help="Translation model name to use.")
    parser.add_argument('--lora_type', 
                        type=lambda x: None if x.lower() == 'none' else x, 
                        default=model_config['lora_type'], 
                        help="LoRA adapter type to use.")
    parser.add_argument('--adapter_name',
                        type=lambda x: None if x.lower() == 'none' else x,
                        default=model_config['adapter_name'],
                        help="Adapter name to use.")
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
                        type=str, 
                        default=infer_config['src_lang'], 
                        help="Source language.")
    parser.add_argument('--tgt_lang', 
                        type=str, 
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
    if args.lora_type is not None:
        print(f"ADAPTER: {args.lora_type.upper()}")
    print(f"INFERENCE: {args.model_type.upper()}")
    if args.data_type == 'sentence':
        print(f"SENTENCE: {args.sentence_text}")
    else:
        print(f"DATASET: {args.dataset_name.upper()}")
    print(f"SOURCE LANG: {args.src_lang.upper()}")
    print(f"TARGET LANG: {args.tgt_lang.upper()}")
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
            df_path = args.dataset
        else:
            df_path = DF_PATH_DICT[args.dataset_name]

        df = pd.read_csv(df_path)
        df = translate_df(translator, df, args)
        df.to_csv(df_path, index=False)

    else:
        raise ValueError("data_type should be either 'sentence' or 'dataset'.")
    

if __name__ == '__main__':
    main()