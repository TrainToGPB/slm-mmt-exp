"""
Perform inference on the specified dataset using the specified pre-trained language model.

The following models are supported:
- opus (제외: 번역 안됨)
- mbart
- nllb-600m
- nllb-1.3b (제외)
- madlad
- mbart-aihub
- llama
- llama-aihub-qlora
- llama-aihub-qlora-bf16 (merged & upscaled)
- llama-aihub-qlora-fp16 (merged & upscaled)
- llama-aihub-qlora-bf16-vllm (merged & upscaled + vLLM)
- llama-aihub-qlora-augment (확장된 데이터)
- llama-aihub-qlora-reverse-new (llama-aihub-qlora 체크포인트에서 새로운 데이터로 한-영 역방향 학습)
- llama-aihub-qlora-reverse-overlap (llama-aihub-qlora 체크포인트에서 동일한 데이터로 한-영 역방향 학습)

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
        'opus': ('Helsinki-NLP/opus-mt-tc-big-en-ko', MarianMTModel, MarianTokenizer),
        'mbart': ('facebook/mbart-large-50-many-to-many-mmt', MBartForConditionalGeneration, MBart50Tokenizer),
        'nllb-600m': ('facebook/nllb-200-distilled-600M', M2M100ForConditionalGeneration, NllbTokenizer),
        'nllb-1.3b': ('facebook/nllb-200-distilled-1.3B', M2M100ForConditionalGeneration, NllbTokenizer),
        'madlad': ('google/madlad400-3b-mt', T5ForConditionalGeneration, T5Tokenizer),
        'mbart-aihub': (os.path.join(SCRIPT_DIR, '../../training/mbart/models/mbart-full'), MBartForConditionalGeneration, MBart50Tokenizer),
        'llama-2-ko-vllm': ('beomi/open-llama-2-ko-7b', None, LlamaTokenizer),
        'llama-2-ko-chat-vllm': ('kfkas/Llama-2-ko-7b-Chat', None, LlamaTokenizer),
        'llama-3-ko-vllm': ('beomi/Llama-3-Open-Ko-8B', None, AutoTokenizer),
        'llama-3-koen-vllm': ('beomi/Llama-3-KoEn-8B', None, AutoTokenizer),
        'llama-3-koen-chat-vllm': ('beomi/Llama-3-KoEn-8B-Instruct-preview', None, AutoTokenizer),
        'llama-2-ko-tiny-qlora': (('beomi/open-llama-2-ko-7b', 'traintogpb/llama-2-en2ko-translator-7b-qlora-adapter'), LlamaForCausalLM, LlamaTokenizer),
        'llama-2-ko-tiny-qlora-bf16': ('traintogpb/llama-2-en2ko-translator-7b-qlora-bf16-upscaled', LlamaForCausalLM, LlamaTokenizer),
        'llama-2-ko-tiny-qlora-fp16': (os.path.join(SCRIPT_DIR, '../../training/llama_qlora/models/baseline-merged-fp16'), LlamaForCausalLM, LlamaTokenizer),
        'llama-2-ko-tiny-qlora-bf16-vllm': ('traintogpb/llama-2-en2ko-translator-7b-qlora-bf16-upscaled', None, LlamaTokenizer),
        'llama-2-ko-tiny-qlora-augment': (('beomi/open-llama-2-ko-7b', os.path.join(SCRIPT_DIR, '../../training/llama_qlora/models/augment')), LlamaForCausalLM, LlamaTokenizer),
        'llama-2-ko-tiny-qlora-reverse-new': (('beomi/open-llama-2-ko-7b', os.path.join(SCRIPT_DIR, '../../training/llama_qlora/models/continuous-reverse-new')), LlamaForCausalLM, LlamaTokenizer),
        'llama-2-ko-tiny-qlora-reverse-overlap': (('beomi/open-llama-2-ko-7b', os.path.join(SCRIPT_DIR, '../../training/llama_qlora/models/continuous-reverse-overlap')), LlamaForCausalLM, LlamaTokenizer),
        'madlad-aihub-7b-bt-qlora': (('google/madlad400-7b-mt-bt', os.path.join(SCRIPT_DIR, '../../training/madlad/models/7b-bt-en2ko')), T5ForConditionalGeneration, T5Tokenizer),
        'mt5-aihub-base-fft': (os.path.join(SCRIPT_DIR, '../../training/mt5/models/base-fft-en2ko-separate-token-constlr-70epoch'), MT5ForConditionalGeneration, T5Tokenizer),
        # alma-qlora-dpo 모델 정상 사용 불가: lora_A 가중치 비어있음
        'alma-qlora-dpo-policy': (('haoranxu/ALMA-7B-Pretrain', os.path.join(SCRIPT_DIR, '../../training/llama_qlora/models/alma-dpo/policy')), LlamaForCausalLM, LlamaTokenizer),
        'alma-qlora-dpo-reference': (('haoranxu/ALMA-7B-Pretrain', os.path.join(SCRIPT_DIR, '../../training/llama_qlora/models/alma-dpo/reference')), LlamaForCausalLM, LlamaTokenizer),
        'llama-2-ko-sparta-tiny-qlora': (('beomi/open-llama-2-ko-7b', 'traintogpb/llama-2-enko-translator-7b-qlora-adapter'), LlamaForCausalLM, LlamaTokenizer),
        'llama-2-ko-sparta-tiny-qlora-bf16': ('traintogpb/llama-2-enko-translator-7b-qlora-bf16-upscaled', LlamaForCausalLM, LlamaTokenizer),
        'llama-2-ko-sparta-tiny-qlora-bf16-vllm': ('traintogpb/llama-2-enko-translator-7b-qlora-bf16-upscaled', None, LlamaTokenizer),
        'llama-2-ko-sparta-mini-qlora': (('beomi/open-llama-2-ko-7b', os.path.join(SCRIPT_DIR, '../../training/llama_qlora/models/llama2-ko-sparta-mini')), LlamaForCausalLM, LlamaTokenizer),
        'llama-2-ko-sparta-mini-qlora-bf16': (os.path.join(SCRIPT_DIR, '../../training/llama_qlora/models/llama2-ko-sparta-mini-merged-bf16'), LlamaForCausalLM, LlamaTokenizer),
        'llama-2-ko-sparta-mini-qlora-bf16-vllm': (os.path.join(SCRIPT_DIR, '../../training/llama_qlora/models/llama2-ko-sparta-mini-merged-bf16'), None, LlamaTokenizer),
        'llama-3-ko-sparta-tiny-qlora': (('beomi/Llama-3-Open-Ko-8B', os.path.join(SCRIPT_DIR, '../../training/llama_qlora/models/llama3-ko-sparta-tiny')), AutoModelForCausalLM, AutoTokenizer),
        'llama-3-ko-sparta-tiny-qlora-bf16': (os.path.join(SCRIPT_DIR, '../../training/llama_qlora/models/llama3-ko-sparta-tiny-merged-bf16'), AutoModelForCausalLM, AutoTokenizer),
        'llama-3-ko-sparta-tiny-qlora-bf16-vllm': (os.path.join(SCRIPT_DIR, '../../training/llama_qlora/models/llama3-ko-sparta-tiny-merged-bf16'), None, AutoTokenizer),
        'llama-3-ko-sparta-tiny-odd-qlora-bf16-vllm': (os.path.join(SCRIPT_DIR, '../../training/llama_qlora/models/llama3-ko-sparta-tiny-odd-merged-bf16'), None, AutoTokenizer),
        'llama-3-koen-sparta-tiny-qlora-bf16-vllm': (os.path.join(SCRIPT_DIR, '../../training/llama_qlora/models/llama3-koen-sparta-tiny-merged-bf16'), None, AutoTokenizer),
        'llama-3-ko-tiny-qlora-bf16-vllm': (os.path.join(SCRIPT_DIR, '../../training/llama_qlora/models/llama3-ko-tiny-merged-bf16'), None, AutoTokenizer),
        'llama-3-ko-mini-qlora-bf16-vllm': (os.path.join(SCRIPT_DIR, '../../training/llama_qlora/models/llama3-ko-mini-merged-bf16'), None, AutoTokenizer),
        'llama-3-ko-sparta-mini-qlora': (('beomi/Llama-3-Open-Ko-8B', os.path.join(SCRIPT_DIR, '../../training/llama_qlora/models/llama3-ko-sparta-mini')), AutoModelForCausalLM, AutoTokenizer),
        'llama-3-ko-sparta-mini-qlora-bf16': (os.path.join(SCRIPT_DIR, '../../training/llama_qlora/models/llama3-ko-sparta-mini-merged-bf16'), AutoModelForCausalLM, AutoTokenizer),
        'llama-3-ko-sparta-mini-qlora-bf16-vllm': (os.path.join(SCRIPT_DIR, '../../training/llama_qlora/models/llama3-ko-sparta-mini-merged-bf16'), None, AutoTokenizer),
        'llama-3-ko-sparta-mini-word-firstlow-all-bidir-qlora-bf16-vllm': (os.path.join(SCRIPT_DIR, '../../training/llama_qlora/models/llama3-word-firstlow-all-bidir-merged-bf16'), None, AutoTokenizer),
        'llama-3-ko-sparta-mini-word-lastlow-all-bidir-qlora-bf16-vllm': (os.path.join(SCRIPT_DIR, '../../training/llama_qlora/models/llama3-word-lastlow-all-bidir-merged-bf16'), None, AutoTokenizer),
        'llama-3-ko-sparta-mini-word-mix-all-bidir-qlora-bf16-vllm': (os.path.join(SCRIPT_DIR, '../../training/llama_qlora/models/llama3-word-mix-all-bidir-merged-bf16'), None, AutoTokenizer),
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
            attn_implementation='flash_attention_2' if 'sparta' in model_type or model_type.startswith('alma') else None,
            torch_dtype=torch_dtype
        )
        model = PeftModel.from_pretrained(
            model, 
            adapter_path, 
            torch_dtype=torch_dtype,
            adapter_name=model_type.split('-')[-1] if 'dpo' in model_type else 'default'
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
        # if 'sparta' in model_type:
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
            'sparta-api': os.path.join(SCRIPT_DIR, "../results/sparta/test_sparta_bidir_api_inferenced.csv"),
            'sparta-llama2': os.path.join(SCRIPT_DIR, "../results/sparta/test_sparta_bidir_llama2_inferenced.csv"),
            'sparta-llama3': os.path.join(SCRIPT_DIR, "../results/sparta/test_sparta_bidir_llama3_inferenced.csv"),
            'dpo': os.path.join(SCRIPT_DIR, "../results/koen_dpo_bidir_inferenced.csv"),
            'ko_words': os.path.join(SCRIPT_DIR, "../results/words/ko_word_test_1k.csv"),
            'en_words': os.path.join(SCRIPT_DIR, "../results/words/en_word_test_1k.csv"),
        }
        file_path = file_path_dict[dataset]

    model_type_dict = {
        'mbart': 'mbart',
        'nllb': 'nllb-600m',
        'madlad': 'madlad',
        'madlad-qlora': 'madlad-aihub-7b-bt-qlora',
        'mt5-fft': 'mt5-aihub-base-fft',
        'alma-pol': 'alma-qlora-dpo-policy',
        'alma-ref': 'alma-qlora-dpo-reference',
        'llama-2-vllm': 'llama-2-ko-vllm',
        'llama-2-chat-vllm': 'llama-2-ko-chat-vllm',
        'llama-3-vllm': 'llama-3-ko-vllm',
        'llama-3-koen-vllm': 'llama-3-koen-vllm',
        'llama-3-koen-chat-vllm': 'llama-3-koen-chat-vllm',
        'llama-2-tiny-qlora': 'llama-2-ko-tiny-qlora',
        'llama-2-tiny-bf16': 'llama-2-ko-tiny-qlora-bf16',
        'llama-2-tiny-vllm': 'llama-2-ko-tiny-qlora-bf16-vllm',
        'llama-2-sparta-tiny-qlora': 'llama-2-ko-sparta-tiny-qlora',
        'llama-2-sparta-tiny-bf16': 'llama-2-ko-sparta-tiny-qlora-bf16',
        'llama-2-sparta-tiny-vllm': 'llama-2-ko-sparta-tiny-qlora-bf16-vllm',
        'llama-2-sparta-mini-qlora': 'llama-2-ko-sparta-mini-qlora',
        'llama-2-sparta-mini-bf16': 'llama-2-ko-sparta-mini-qlora-bf16',
        'llama-2-sparta-mini-vllm': 'llama-2-ko-sparta-mini-qlora-bf16-vllm',
        'llama-3-sparta-tiny-qlora': 'llama-3-ko-sparta-tiny-qlora',
        'llama-3-sparta-tiny-bf16': 'llama-3-ko-sparta-tiny-qlora-bf16',
        'llama-3-sparta-tiny-vllm': 'llama-3-ko-sparta-tiny-qlora-bf16-vllm',
        'llama-3-koen-sparta-tiny-vllm': 'llama-3-koen-sparta-tiny-qlora-bf16-vllm',
        'llama-3-sparta-tiny-odd-vllm': 'llama-3-ko-sparta-tiny-odd-qlora-bf16-vllm',
        'llama-3-tiny-vllm': 'llama-3-ko-tiny-qlora-bf16-vllm',
        'llama-3-mini-vllm': 'llama-3-ko-mini-qlora-bf16-vllm',
        'llama-3-sparta-mini-qlora': 'llama-3-ko-sparta-mini-qlora',
        'llama-3-sparta-mini-bf16': 'llama-3-ko-sparta-mini-qlora-bf16',
        'llama-3-sparta-mini-vllm': 'llama-3-ko-sparta-mini-qlora-bf16-vllm',
        'llama-3-firstlow-all-bidir-vllm': 'llama-3-ko-sparta-mini-word-firstlow-all-bidir-qlora-bf16-vllm',
        'llama-3-lastlow-all-bidir-vllm': 'llama-3-ko-sparta-mini-word-lastlow-all-bidir-qlora-bf16-vllm',
        'llama-3-mix-all-bidir-vllm': 'llama-3-ko-sparta-mini-word-mix-all-bidir-qlora-bf16-vllm',
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
        source_column = "src" if any(args.dataset.startswith(bidir_data) for bidir_data in ['sparta', 'sample', 'dpo']) else args.src_lang
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
