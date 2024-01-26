# built-in
import random
from tqdm import tqdm

# third-party
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import MarianMTModel, MarianTokenizer
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from transformers import M2M100ForConditionalGeneration, NllbTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import BitsAndBytesConfig
from peft import PeftModel


def set_seed(SEED=42):
    """
    Set the random seeds for reproducibility in a PyTorch environment.

    Parameters:
    - SEED (int, optional): Seed value to be used for random number generation. Default is 42.

    Usage:
    Call this function before running any code that involves random number generation to ensure reproducibility.

    Example:
    set_seed(123)
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def load_model_and_tokenizer(model_type):
    """
    Load a pre-trained language model and tokenizer based on the specified model type.

    Parameters:
    - model_type (str): Type of the pre-trained language model.

    Returns:
    - model (PreTrainedModel): Loaded pre-trained language model.
    - tokenizer (PreTrainedTokenizer): Loaded tokenizer.
    """
    assert model_type in ['opus', 'mbart', 'nllb-600m', 'nllb-1.3b', 'madlad', 'mbart-aihub', 'llama-aihub-qlora'], 'Wrong model type'

    # existing translation models
    if model_type == 'opus':
        model_name = 'Helsinki-NLP/opus-mt-tc-big-en-ko'
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
    elif model_type == 'mbart':
        model_name = 'facebook/mbart-large-50-many-to-many-mmt'
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = MBart50Tokenizer.from_pretrained(model_name)
    elif model_type == 'nllb-600m':
        model_name = 'facebook/nllb-200-distilled-600M'
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = NllbTokenizer.from_pretrained(model_name)
    elif model_type == 'nllb-1.3b':
        model_name = 'facebook/nllb-200-distilled-1.3B'
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = NllbTokenizer.from_pretrained(model_name)
    elif model_type == 'madlad':
        model_name = 'google/madlad400-3b-mt'
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    # finetuned translation models
    elif model_type == 'mbart-aihub':
        state_dict_path = '../../training/mbart/models/mbart-baseline-merged.pth'
        plm_name = 'facebook/mbart-large-50'
        model = AutoModelForSeq2SeqLM.from_pretrained(plm_name, state_dict=torch.load(state_dict_path))
        tokenizer = AutoTokenizer.from_pretrained(plm_name)
    elif model_type == 'llama-aihub-qlora':
        plm_name = 'beomi/open-llama-2-ko-7b'
        lora_path = '../../training/llama_qlora/models/sft' # HuggingFace에 업로드 필요
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        torch_dtype = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(
            plm_name, 
            quantization_config=bnb_config, 
            torch_dtype=torch_dtype
        )
        model = PeftModel.from_pretrained(
            model, 
            lora_path, 
            torch_dtype=torch_dtype
        )

        tokenizer = AutoTokenizer.from_pretrained(plm_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        tokenizer.model_max_length = 768

    return model, tokenizer


def translate(model, tokenizer, text, model_type, device, max_length=512):
    """
    Translate a given text using the specified pre-trained language model.

    Parameters:
    - model (PreTrainedModel): Pre-trained language model.
    - tokenizer (PreTrainedTokenizer): Tokenizer for encoding the input.
    - text (str): Input text to be translated.
    - model_type (str): Type of the pre-trained language model.
    - device (str): Device for model inference (e.g., 'cuda' for GPU).
    - max_length (int): Maximum length of the translated sequence.

    Returns:
    - str: Translated text.
    """
    model.to(device)

    if model_type == 'madlad':
        text = ' '.join(['<2ko>', text])
    elif model_type == 'llama-aihub-qlora':
        text = ' '.join(['### English:', text, '### 한국어:'])

    if model_type == 'mbart' or model_type == 'mbart-aihub':
        tokenizer.src_lang = 'en_XX'
    elif model_type == 'nllb-600m' or model_type == 'nllb-1.3b':
        tokenizer.src_lang = 'eng_Latn'

    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    if model_type == 'opus' or model_type == 'madlad' or model_type == 'llama-aihub-qlora':
        outputs = model.generate(**inputs, max_length=max_length)
    elif model_type == 'mbart' or model_type == 'mbart-aihub':
        outputs = model.generate(**inputs, max_length=max_length, forced_bos_token_id=tokenizer.lang_code_to_id['ko_KR'])
    elif model_type == 'nllb-600m' or model_type == 'nllb-1.3b':
        outputs = model.generate(**inputs, max_length=max_length, forced_bos_token_id=tokenizer.lang_code_to_id['kor_Hang'])
    
    translated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    return translated_text


def inference(model_type, source_column, target_column, eval_path, output_path, device):
    """
    Perform inference on a dataset using the specified pre-trained language model.

    Parameters:
    - model_type (str): Type of the pre-trained language model.
    - source_column (str): Column name containing source texts in the evaluation dataset.
    - target_column (str): Column name to store translated texts in the output dataset.
    - eval_path (str): File path of the evaluation dataset in CSV format.
    - output_path (str): File path to save the translated output in CSV format.
    - device (str): Device for model inference (e.g., 'cuda' for GPU).
    """
    set_seed(42)
    model, tokenizer = load_model_and_tokenizer(model_type)
    model.to(device)

    tqdm.pandas(desc="Translating")

    eval_df = pd.read_csv(eval_path)
    eval_df[target_column] = eval_df[source_column].progress_apply(lambda text: translate(model, tokenizer, text, model_type, device))
    eval_df.to_csv(output_path, index=False)


def inference_single(model_type, text, device):
    """
    Perform inference on a single text using the specified pre-trained language model.

    Parameters:
    - model_type (str): Type of the pre-trained language model.
    - text (str): Input text to be translated.
    - device (str): Device for model inference (e.g., 'cuda' for GPU).

    Returns:
    - str: Translated text.
    """
    set_seed(42)
    model, tokenizer = load_model_and_tokenizer(model_type)
    model.to(device)

    translation = translate(model, tokenizer, text, model_type, device)

    return translation


if __name__ == '__main__':
    import argparse
    """
    [model_type]
    - mbart
    - nllb-600m
    - madlad
    - mbart-aihub (X)
    - llama-aihub-qlora (X)
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", DEVICE)

    parser = argparse.ArgumentParser(description='Inference Script')
    parser.add_argument(
        '--model_type', 
        type=str, 
        choices=['opus', 'mbart', 'nllb-600m', 'nllb-1.3b', 'madlad'], # mbart-aihub, llama, llama-quant, llama-aihub-qlora 추가 예정
        default='mbart', 
        help='Type of the model to use for inference'
    )
    parser.add_argument(
        '--en_text', 
        type=str, 
        default="NMIXX is a South Korean girl group that made a comeback on January 15, 2024 with their new song 'DASH'.",
        help='English text to be translated'
    )

    args = parser.parse_args()
    print(f"Inference model: {args.model_type.upper()}")

    # inference sentence
    print(f"원본 영어 문장: {args.en_text}")
    print(f"번역 한글 문장: {inference_single(args.model_type, args.en_text, DEVICE)}")
