# built-in
import os
import sys

# third-party
import torch
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from transformers import M2M100ForConditionalGeneration, NllbTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer

# custom
sys.path.append('../../../')
from custom_utils.training_utils import set_seed


def load_model_and_tokenizer(model_type):
    assert model_type in ['opus', 'mbart', 'nllb', 'madlad'], 'Error: Wrong Model Type'

    if model_type == 'opus':
        model_name = 'Helsinki-NLP/opus-mt-tc-big-en-ko'
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
    elif model_type == 'mbart':
        model_name = 'facebook/mbart-large-50-many-to-many-mmt'
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = MBart50Tokenizer.from_pretrained(model_name)
    elif model_type == 'nllb':
        model_name = 'facebook/nllb-200-distilled-600M'
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = NllbTokenizer.from_pretrained(model_name)
    elif model_type == 'madlad':
        model_name = 'google/madlad400-3b-mt'
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    return model, tokenizer


def translate(model, tokenizer, text, model_type, device, max_length=512):
    model.to(device)

    if model_type == 'madlad':
        text = '<2ko> ' + text

    if model_type == 'mbart':
        tokenizer.src_lang = 'en_XX'
    elif model_type == 'nllb':
        tokenizer.src_lang = 'eng_Latn'

    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    if model_type == 'opus' or model_type == 'madlad':
        outputs = model.generate(**inputs)
    elif model_type == 'mbart':
        outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id['ko_KR'])
    elif model_type == 'nllb':
        outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id['kor_Hang'])

    translated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    return translated_text


def inference(model_type, source_column, target_column, eval_path, output_path, device):
    set_seed(42)
    model, tokenizer = load_model_and_tokenizer(model_type)
    model.to(device)

    eval_df = pd.read_csv(eval_path)
    eval_df[target_column] = eval_df[source_column].apply(lambda text: translate(model, tokenizer, text, model_type, device))
    eval_df.to_csv(output_path, index=False)


if __name__ == '__main__':
    """
    [MODEL_TYPE]
    - opus
    - mbart
    - nllb
    - madlad
    """
    MODEL_TYPE = "madlad"
    SOURCE_COLUMN = "en"
    TARGET_COLUMN = MODEL_TYPE + "_trans"
    EVAL_PATH = "../results/test_tiny_uniform100_inferenced.csv"
    OUTPUT_PATH = "../results/test_tiny_uniform100_inferenced.csv"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", DEVICE)

    inference(MODEL_TYPE, SOURCE_COLUMN, TARGET_COLUMN, EVAL_PATH, OUTPUT_PATH, DEVICE)
