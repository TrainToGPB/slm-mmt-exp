# built-in
import os
import sys
from pathlib import Path

# third-party
import torch
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import MarianMTModel, MarianTokenizer
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from transformers import M2M100ForConditionalGeneration, NllbTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer

# custom
sys.path.append('../../../')
from custom_utils.training_utils import set_seed


def load_model_and_tokenizer(model_type):
    """
    Load the pre-trained translation model and tokenizer based on the specified model type.

    Parameters:
    - model_type (str): Type of the translation model ('opus', 'mbart', 'nllb-600m', 'nllb-1.3b', 'madlad').

    Returns:
    - model (PreTrainedModel): Loaded translation model.
    - tokenizer (PreTrainedTokenizer): Loaded tokenizer corresponding to the model.
    """
    assert model_type in ['opus', 'mbart', 'nllb-600m', 'nllb-1.3b', 'madlad', 'mbart-aihub'], 'Wrong model type'

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
    
    return model, tokenizer


def translate(model, tokenizer, text, model_type, device, max_length=512):
    """
    Translate text from English to Korean using the specified translation model and tokenizer.

    Parameters:
    - model (PreTrainedModel): Loaded translation model.
    - tokenizer (PreTrainedTokenizer): Loaded tokenizer corresponding to the model.
    - text (str): English text to be translated.
    - model_type (str): Type of the translation model ('opus', 'mbart', 'nllb-600m', 'nllb-1.3b', 'madlad').
    - device (str): Device to use for inference ('cpu' or 'cuda').
    - max_length (int, optional): Maximum length of the translated text. Default is 512.

    Returns:
    - translated_text (str): Translated Korean text.
    """
    model.to(device)

    if model_type == 'madlad':
        text = '<2ko> ' + text

    if model_type == 'mbart' or model_type == 'mbart-aihub':
        tokenizer.src_lang = 'en_XX'
    elif model_type == 'nllb-600m' or model_type == 'nllb-1.3b':
        tokenizer.src_lang = 'eng_Latn'

    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    if model_type == 'opus' or model_type == 'madlad':
        outputs = model.generate(**inputs, max_length=max_length)
    elif model_type == 'mbart' or model_type == 'mbart-aihub':
        outputs = model.generate(**inputs, max_length=max_length, forced_bos_token_id=tokenizer.lang_code_to_id['ko_KR'])
    elif model_type == 'nllb-600m' or model_type == 'nllb-1.3b':
        outputs = model.generate(**inputs, max_length=max_length, forced_bos_token_id=tokenizer.lang_code_to_id['kor_Hang'])

    translated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    return translated_text


def inference(model_type, source_column, target_column, eval_path, output_path, device):
    """
    Perform inference using the specified translation model on a DataFrame and save the results to a CSV file.

    Parameters:
    - model_type (str): Type of the translation model ('opus', 'mbart', 'nllb-600m', 'nllb-1.3b', 'madlad').
    - source_column (str): Name of the column containing English text in the DataFrame.
    - target_column (str): Name of the column to store the translated Korean text in the DataFrame.
    - eval_path (str): Path to the CSV file containing the evaluation DataFrame.
    - output_path (str): Path to save the output CSV file with translated text.
    - device (str): Device to use for inference ('cpu' or 'cuda').

    Returns:
    - None: The results are saved to the specified output_path.
    """
    set_seed(42)
    model, tokenizer = load_model_and_tokenizer(model_type)
    model.to(device)

    eval_df = pd.read_csv(eval_path)
    eval_df[target_column] = eval_df[source_column].apply(lambda text: translate(model, tokenizer, text, model_type, device))
    eval_df.to_csv(output_path, index=False)


def inference_single(model_type, text, device):
    """
    Perform inference using the specified translation model on a DataFrame and save the results to a CSV file.

    Parameters:
    - model_type (str): Type of the translation model ('opus', 'mbart', 'nllb-600m', 'nllb-1.3b', 'madlad').
    - source_column (str): Name of the column containing English text in the DataFrame.
    - target_column (str): Name of the column to store the translated Korean text in the DataFrame.
    - eval_path (str): Path to the CSV file containing the evaluation DataFrame.
    - output_path (str): Path to save the output CSV file with translated text.
    - device (str): Device to use for inference ('cpu' or 'cuda').

    Returns:
    - None: The results are saved to the specified output_path.
    """
    set_seed(42)
    model, tokenizer = load_model_and_tokenizer(model_type)
    model.to(device)

    translation = translate(model, tokenizer, text, model_type, device)

    return translation


if __name__ == '__main__':
    """
    [MODEL_TYPE]
    - opus
    - mbart
    - nllb-600m
    - nllb-1.3b
    - madlad
    - mbart-aihub
    """
    SOURCE_COLUMN = "en"
    EVAL_PATH = "../results/test_tiny_uniform100_inferenced.csv"
    OUTPUT_PATH = "../results/test_tiny_uniform100_inferenced.csv"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", DEVICE)

    model_types = ['mbart-aihub']
    for MODEL_TYPE in model_types:
        TARGET_COLUMN = MODEL_TYPE + "_trans"
        # inference(MODEL_TYPE, SOURCE_COLUMN, TARGET_COLUMN, EVAL_PATH, OUTPUT_PATH, DEVICE)

        TEXT_KO = "엔믹스는 한국의 걸그룹으로, 2024년 1월 15일 신곡 'DASH'로 컴백했습니다."
        TEXT_EN = "NMIXX is a South Korean girl group that made a comeback on January 15, 2024 with their new song 'DASH'."
        translation = inference_single(MODEL_TYPE, TEXT_EN, DEVICE)
        print(translation)
