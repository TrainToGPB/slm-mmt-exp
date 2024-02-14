# built-in
import os
import re
import sys
import argparse
from tqdm import tqdm

# third-party
import torch
import deepspeed
import pandas as pd
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import MarianMTModel, MarianTokenizer
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from transformers import M2M100ForConditionalGeneration, NllbTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import BitsAndBytesConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# custom
sys.path.append('../../../')
from custom_utils.training_utils import set_seed


MASTER_PORT = 60001
os.environ['MASTER_ADDR'] = "localhost"
os.environ['MASTER_PORT'] = str(MASTER_PORT)
os.environ['RANK'] = "0"
os.environ['LOCAL_RANK'] = "-1"

NUM_GPUS = 1
if NUM_GPUS == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    os.environ['WORLD_SIZE'] = "1"
    WORLD_SIZE = 1
elif NUM_GPUS == 2:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    os.environ['WORLD_SIZE'] = "2"
    WORLD_SIZE = 2
elif NUM_GPUS == 4:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    os.environ['WORLD_SIZE'] = "4"
    WORLD_SIZE = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)


def load_model_and_tokenizer(model_type):
    """
    Load a pre-trained language model and tokenizer based on the specified model type.

    Parameters:
    - model_type (str): Type of the pre-trained language model.

    Returns:
    - model (PreTrainedModel): Loaded pre-trained language model.
    - tokenizer (PreTrainedTokenizer): Loaded tokenizer.
    """
    # Define model and tokenizer mapping
    model_mapping = {
        'opus': ('Helsinki-NLP/opus-mt-tc-big-en-ko', MarianMTModel, MarianTokenizer),
        'mbart': ('facebook/mbart-large-50-many-to-many-mmt', MBartForConditionalGeneration, MBart50Tokenizer),
        'nllb-600m': ('facebook/nllb-200-distilled-600M', M2M100ForConditionalGeneration, NllbTokenizer),
        'nllb-1.3b': ('facebook/nllb-200-distilled-1.3B', M2M100ForConditionalGeneration, NllbTokenizer),
        'madlad': ('google/madlad400-3b-mt', T5ForConditionalGeneration, T5Tokenizer),
        'mbart-aihub': ('../../training/mbart/models/mbart-full', MBartForConditionalGeneration, MBart50Tokenizer),
        'llama': ('beomi/open-llama-2-ko-7b', LlamaForCausalLM, LlamaTokenizer),
        'llama-aihub-qlora': (('beomi/open-llama-2-ko-7b', '../../training/llama_qlora/models/baseline'), LlamaForCausalLM, LlamaTokenizer),
        'llama-aihub-qlora-bf16': ('../../training/llama_qlora/models/baseline-merged-bf16', LlamaForCausalLM, LlamaTokenizer),
        'llama-aihub-qlora-fp16': ('../../training/llama_qlora/models/baseline-merged-fp16', LlamaForCausalLM, LlamaTokenizer),
        'llama-aihub-qlora-fp16-ds': ('../../training/llama_qlora/models/baseline-merged-fp16', LlamaForCausalLM, LlamaTokenizer),
        'llama-aihub-qlora-augment': (('beomi/open-llama-2-ko-7b', '../../training/llama_qlora/models/augment'), LlamaForCausalLM, LlamaTokenizer),
        'llama-aihub-qlora-reverse-new': (('beomi/open-llama-2-ko-7b', '../../training/llama_qlora/models/continuous-reverse-new'), LlamaForCausalLM, LlamaTokenizer),
        'llama-aihub-qlora-reverse-overlap': (('beomi/open-llama-2-ko-7b', '../../training/llama_qlora/models/continuous-reverse-overlap'), LlamaForCausalLM, LlamaTokenizer),
    }
    assert model_type in model_mapping, 'Wrong model type'

    # Load model and tokenizer based on the model type
    model_name, model_cls, tokenizer_cls = model_mapping[model_type]
    if isinstance(model_name, tuple):
        model_name, adapter_path = model_name[0], model_name[1]

    if model_type.startswith('llama-aihub-qlora'):
        # Special configurations for llama-aihub-qlora models
        if '16' in model_type:
            model = model_cls.from_pretrained(model_name)
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
            torch_dtype = torch.bfloat16
            model = LlamaForCausalLM.from_pretrained(
                model_name, 
                quantization_config=bnb_config, 
                torch_dtype=torch_dtype
            )
            print(adapter_path)
            model = PeftModel.from_pretrained(
                model, 
                adapter_path, 
                torch_dtype=torch_dtype
            )

        tokenizer = tokenizer_cls.from_pretrained(model_name)
        tokenizer.pad_token = "</s>"
        tokenizer.pad_token_id = 2
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.eos_token_id = 46332
        tokenizer.add_eos_token = True
        tokenizer.padding_side = 'right'
        tokenizer.model_max_length = 768

        if model_type.endswith('ds'):
            deepspeed.init_distributed(
                dist_backend='nccl',
                distributed_port=MASTER_PORT,
                rank=0,
                world_size=WORLD_SIZE
            )
            ds_engine = deepspeed.init_inference(
                model,
                tensor_parallel={'tp_size': WORLD_SIZE},
                # mp_size=WORLD_SIZE,
                dtype=torch.half,
                checkpoint=None,
                # injection_policy={LlamaDecoderLayer: ('self_attn.o_proj', 'mlp.down_proj')},
                replace_method='auto',
                replace_with_kernel_inject=True
            )
            model = ds_engine.module

    else:
        # For other models, simply load from pretrained
        model = model_cls.from_pretrained(model_name)
        tokenizer = tokenizer_cls.from_pretrained(model_name)

    return model, tokenizer


def translate(model, tokenizer, text, model_type, print_result=False, max_length=512):
    """
    Translate a given text using the specified pre-trained language model.

    Parameters:
    - model (PreTrainedModel): Pre-trained language model.
    - tokenizer (PreTrainedTokenizer): Tokenizer for encoding the input.
    - text (str): Input text to be translated.
    - model_type (str): Type of the pre-trained language model.
    - max_length (int): Maximum length of the translated sequence.

    Returns:
    - str: Translated text.
    """
    model.to(DEVICE)

    if model_type == 'madlad':
        text = ' '.join(['<2ko>', text])
    elif 'llama' in model_type:
        text = f"### English: {text}\n### 한국어: "

    if 'mbart' in model_type:
        src_lang = 'en_XX'
        tgt_lang = 'ko_KR'
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang
    elif 'nllb' in model_type:
        src_lang = 'eng_Latn'
        tgt_lang = 'kor_Hang'
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang

    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    if model_type.startswith('llama-aihub-qlora'):
        inputs['input_ids'] = inputs['input_ids'][0][:-1].unsqueeze(dim=0)
        inputs['attention_mask'] = inputs['attention_mask'][0][:-1].unsqueeze(dim=0)
        if model_type.endswith('ds'):
            use_cache = False
        else:
            use_cache = None
        outputs = model.generate(**inputs, max_length=max_length, eos_token_id=46332, use_cache=use_cache)
    elif 'mbart' in model_type or 'nllb' in model_type:
        outputs = model.generate(**inputs, max_length=max_length, forced_bos_token_id=tokenizer.lang_code_to_id[tokenizer.tgt_lang])
    else:
        outputs = model.generate(**inputs, max_length=max_length)
    
    input_len = len(inputs['input_ids'].squeeze()) if model_type.startswith('llama') else 0
    
    translated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    translated_text = re.sub(r'\s+', ' ', translated_text)
    translated_text = translated_text.strip()
    
    if print_result:
        print(translated_text)

    return translated_text


def inference(model_type, source_column, target_column, file_path):
    """
    Perform inference on a dataset using the specified pre-trained language model.

    Parameters:
    - model_type (str): Type of the pre-trained language model.
    - source_column (str): Column name containing source texts in the evaluation dataset.
    - target_column (str): Column name to store translated texts in the output dataset.
    - file_path (str): File path of the inference dataset in CSV format.
    """
    set_seed(42)
    model, tokenizer = load_model_and_tokenizer(model_type)
    model.to(DEVICE)

    print_result = True
    max_length = 768 if 'llama' in model_type else 512

    eval_df = pd.read_csv(file_path)
    tqdm.pandas(desc="Translating")
    eval_df[target_column] = eval_df[source_column].progress_apply(lambda text: translate(model, tokenizer, text, model_type, print_result, max_length))
    eval_df.to_csv(file_path, index=False)


def inference_single(model_type, text):
    """
    Perform inference on a single text using the specified pre-trained language model.

    Parameters:
    - model_type (str): Type of the pre-trained language model.
    - text (str): Input text to be translated.

    Returns:
    - str: Translated text.
    """
    set_seed(42)
    model, tokenizer = load_model_and_tokenizer(model_type)
    model.to(DEVICE)

    translation = translate(model, tokenizer, text, model_type, DEVICE)

    return translation


if __name__ == '__main__':
    import argparse
    """
    [MODEL_TYPE]
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
    - llama-aihub-qlora-fp16-ds (merged & upscaled + DeepSpeed)
    - llama-aihub-qlora-augment (확장된 데이터)
    - llama-aihub-qlora-reverse-new (llama-aihub-qlora 체크포인트에서 새로운 데이터로 한-영 역방향 학습)
    - llama-aihub-qlora-reverse-overlap (llama-aihub-qlora 체크포인트에서 동일한 데이터로 한-영 역방향 학습)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='aihub', help="Dataset for inference")
    args = parser.parse_args()
    dataset = args.dataset
    
    source_column = "en"
    if dataset == 'aihub':
        file_path = "../results/test_tiny_uniform100_inferenced.csv"
    elif dataset == 'flores':
        file_path = "../results/test_flores_inferenced.csv"

    model_types = [
        'llama-aihub-qlora-reverse-overlap',
    ]
    for model_type in model_types:
        print(f"Inference model: {model_type.upper()}")

        # inference dataset
        target_column = model_type + "_trans"
        inference(model_type, source_column, target_column, file_path)
        
        # # inference sentence
        # # text_en = "NMIXX is a South Korean girl group that made a comeback on January 15, 2024 with their new song 'DASH'."
        # text_en = 'Kang Hye-in, director of Korean market at GINGKOO, said, "I hope this agreement will be a good opportunity for both companies to lead the Asian blockchain industry."'
        # translation = inference_single(model_type, text_en, DEVICE)
        # print(translation)
