# built-in
import os
import sys

# third-party
import torch
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer

# custom
sys.path.append('../../../')
from custom_utils.training_utils import set_seed


def load_model_and_tokenizer(model_name):
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer


def translate_text(model, tokenizer, text, device, max_length=512):
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    model.to(device)
    outputs = model.generate(**inputs)
    translated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return translated_text


def inference(model_name, source_column, target_column, eval_path, device):
    set_seed(42)

    eval_df = pd.read_csv(eval_path)

    model, tokenizer = load_model_and_tokenizer(model_name)
    model.to(device)

    eval_df[target_column] = eval_df[source_column].apply(lambda text: translate_text(model, tokenizer, text, device))

    output_csv_path = "../results/test_tiny_uniform100_inferenced.csv"
    eval_df.to_csv(output_csv_path, index=False)


if __name__ == '__main__':
    MODEL_NAME = "Helsinki-NLP/opus-mt-tc-big-en-ko"
    SOURCE_COLUMN = "en"
    TARGET_COLUMN = "opus_trans"
    EVAL_PATH = "../results/test_tiny_uniform100_inferenced.csv"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", DEVICE)

    inference(MODEL_NAME, SOURCE_COLUMN, TARGET_COLUMN, EVAL_PATH, DEVICE)
