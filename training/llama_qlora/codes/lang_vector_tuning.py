import os
from tqdm import tqdm

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer


def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    return model


def get_lang_vector(tgt_lang_model_name, en_lang_model_name):
    tgt_lang_model = load_model(tgt_lang_model_name)
    en_lang_model = load_model(en_lang_model_name)

    lang_vector = {}
    tqdm_iterator = tqdm(tgt_lang_model.named_parameters(), total=len(list(tgt_lang_model.named_parameters())), desc="Making lang vector")
    for name, param_tgt in tqdm_iterator:
        param_en = dict(en_lang_model.named_parameters())[name]
        lang_vector[name] = param_tgt - param_en
    return lang_vector


def add_lang_vector(model_name, lang_vector):
    model = load_model(model_name)
    if isinstance(lang_vector, dict):
        lang_vector = transform_lang_vector_to_model(model_name, lang_vector)
    elif isinstance(lang_vector, str):
        lang_vector = load_model(lang_vector)
    else:
        lang_vector = lang_vector

    tqdm_iterator = tqdm(model.named_parameters(), total=len(list(model.named_parameters())), desc="Adding lang vector")
    for name, param in tqdm_iterator:
        if isinstance(lang_vector, dict) and name in lang_vector.keys():
            param.data = param.data + lang_vector[name].data.clone()
        elif hasattr(lang_vector, 'state_dict') and name in lang_vector.state_dict().keys():
            param.data = param.data + lang_vector.state_dict()[name].data.clone()

    return model


def transform_lang_vector_to_model(lang_model_name, lang_vector):
    config = AutoConfig.from_pretrained(lang_model_name)
    lang_vector_model = AutoModelForCausalLM.from_config(config)

    tqdm_iterator = tqdm(lang_vector_model.named_parameters(), total=len(list(lang_vector_model.named_parameters())), desc="Wrapping lang vector")
    for name, param in tqdm_iterator:
        if name in lang_vector.keys():
            param.data = lang_vector[name].data.clone()

    lang_vector_model = lang_vector_model.to(torch.bfloat16)
    
    return lang_vector_model


def save_lang_vector(lang_vector, output_dir, lang_model_name=None):
    if isinstance(lang_vector, dict):
        lang_vector_model = transform_lang_vector_to_model(lang_model_name, lang_vector)
    else:
        lang_vector_model = lang_vector

    training_args = TrainingArguments(output_dir)
    trainer = Trainer(model=lang_vector_model, args=training_args)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)


def get_frobenius_norm_(weight_diff):
    frobenius_norm = 0.0
    tqdm_iterator = tqdm(weight_diff.named_parameters(), total=len(list(weight_diff.named_parameters())), desc="Calculating Frobenius norm")
    for name, param in tqdm_iterator:
        frobenius_norm += torch.norm(param, p='fro').item()
    return frobenius_norm


def get_frobenius_norm(lang_vector_name, add_vector=False):
    if isinstance(lang_vector_name, str):
        lang_vector = load_model(lang_vector_name)
    elif isinstance(lang_vector_name, list):
        if add_vector:
            lang_vector = add_lang_vector(lang_vector_name[0], load_model(lang_vector_name[1]))
        else:
            lang_vector_dict = get_lang_vector(lang_vector_name[0], lang_vector_name[1])
            lang_vector = transform_lang_vector_to_model(lang_vector_name[0], lang_vector_dict)
    frobenius_norm = get_frobenius_norm_(lang_vector)
    return frobenius_norm


def make_lang_vector(tgt_lang='ko'):
    en_model_name = 'meta-llama/Meta-Llama-3-8B'
    tgt_lang_models_dict = {
        'ko': 'beomi/Llama-3-Open-Ko-8B',
        'ja': 'rinna/llama-3-youko-8b',
        'zh': 'hfl/llama-3-chinese-8b',
    }
    tgt_lang_model_name = tgt_lang_models_dict[tgt_lang]

    lang_vector = get_lang_vector(tgt_lang_model_name, en_model_name)

    output_dir = f'../models/llama3-lang-vector-{tgt_lang}'
    save_lang_vector(tgt_lang_model_name, lang_vector, output_dir)


def calculate_frobenius_norm(lang1='ko', lang2=None, add_vector=False):
    if lang2 is None:
        weight_diff = f'../models/llama3-lang-vector-{lang1}'
    else:
        weight_diff = [f'../models/llama3-lang-vector-{lang1}', f'../models/llama3-lang-vector-{lang2}']

    frobenius_norm = get_frobenius_norm(weight_diff, add_vector=add_vector)

    if lang2 is None:
        print(f"Frobenius norm of the {lang1.upper()} lang vector: {frobenius_norm:.3f}")
    else:
        if add_vector:
            print(f"Frobenius norm of the {lang1.upper()} + {lang2.upper()} lang vectors: {frobenius_norm:.3f}")
        else:
            print(f"Frobenius norm of the {lang1.upper()} & {lang2.upper()} lang vectors: {frobenius_norm:.3f}")


def save_multilingual_model(src_lang='ko', tgt_lang='ja'):
    src_lang_models_dict = {
        'ko': 'beomi/Llama-3-Open-Ko-8B',
        'ja': 'rinna/llama-3-youko-8b',
        'zh': 'hfl/llama-3-chinese-8b',
        'koja': '../models/llama3-multilingual-koja-langvec',
        'kozh': '../models/llama3-multilingual-kozh-langvec',
    }
    src_model_name = src_lang_models_dict[src_lang]
    lang_vector_name = f'../models/llama3-lang-vector-{tgt_lang}'
    lang_vector = load_model(lang_vector_name)

    multilang_model = add_lang_vector(src_model_name, lang_vector)

    output_dir = f'../models/llama3-multilingual-{src_lang}{tgt_lang}-langvec'
    save_lang_vector(multilang_model, output_dir)


def main():
    # make_lang_vector(tgt_lang='ko')
    # calculate_frobenius_norm(lang1='ko', lang2='ja', add_vector=True)
    save_multilingual_model(src_lang='ko', tgt_lang='zh')


if __name__ == '__main__':
    main()
