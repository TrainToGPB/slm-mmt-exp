import yaml
from tqdm import tqdm

import pandas as pd


def read_eval_dict_yaml(file_path):
    try:
        with open(file_path, 'r') as file:
            eval_dict = yaml.load(file, Loader=yaml.FullLoader)
    except FileNotFoundError:
        eval_dict = {}
    return eval_dict


def save_eval_dict_yaml(eval_dict, save_path):
    with open(save_path, 'w') as file:
        yaml.dump(eval_dict, file)


def check_linebreaks(src_text, trans_text, infer_type):
    src_linebreaks = src_text.count('\n')
    trans_linebreaks = trans_text.count('\n')
    if infer_type == 'reflect':
        followed = None
        pass
    elif infer_type == 'ignore':
        followed = None
        pass
    else:
        raise ValueError(f'Invalid infer_type: {infer_type}')
    return followed


def check_propernouns(tgt_text, trans_text, infer_type):
    tgt_propernouns = set([word for word in tgt_text.split() if word.istitle()])

