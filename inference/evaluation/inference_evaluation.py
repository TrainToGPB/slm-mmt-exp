import yaml
from tqdm import tqdm

import evaluate
import pandas as pd
from transformers import AutoTokenizer
from comet import download_model, load_from_checkpoint


def read_eval_dict_yaml(file_path):
    try:
        with open(file_path, 'r') as file:
            eval_dict = yaml.load(file, Loader=yaml.FullLoader)
    except FileNotFoundError:
        eval_dict = dict()
    return eval_dict


def save_eval_dict_yaml(eval_dict, save_path):
    with open(save_path, 'w') as file:
        yaml.dump(eval_dict, file)


def rearrange_columns(df, front_cols=['tgt'], back_cols=['src', 'direction', 'data_source']):
    other_cols = [col for col in df.columns if col not in front_cols + back_cols]
    df = df[front_cols + other_cols + back_cols]
    return df


def load_xcomet_model():
    model_path = download_model("Unbabel/XCOMET-XL")
    model = load_from_checkpoint(model_path)
    return model


def calculate_xcomet(eval_df, model, tgt_col, src_col='en', ref_col='ko'):
    triplets = []
    tqdm_iterator = tqdm(eval_df.iterrows(), total=len(eval_df), desc="Preparing triplets")
    for _, example in tqdm_iterator:
        src_text = example[src_col]
        tgt_text = example[tgt_col] if not pd.isna(example[tgt_col]) else ' '
        ref_text = example[ref_col] if ref_col is not None else None

        if ref_text is not None:
            triplet = {"src": src_text, "mt": tgt_text, "ref": ref_text}
        else:
            triplet = {"src": src_text, "mt": tgt_text}
        triplets.append(triplet)

    model_output = model.predict(triplets, batch_size=128, gpus=1)
    score = sum(model_output["scores"]) / len(model_output["scores"]) * 100

    return score


def calculate_xcomet_line_by_line(eval_df, model, tgt_col, src_col='en', ref_col='ko'):
    triplets = []
    tqdm_iterator = tqdm(eval_df.iterrows(), total=len(eval_df), desc="Preparing triplets")
    for _, example in tqdm_iterator:
        src_text = example[src_col]
        tgt_text = example[tgt_col] if not pd.isna(example[tgt_col]) else ' '

        if ref_col is not None:
            ref_text = example[ref_col]
            triplet = {"src": src_text, "mt": tgt_text, "ref": ref_text}
        else:
            triplet = {"src": src_text, "mt": tgt_text}
        triplets.append(triplet)

    model_output = model.predict(triplets, batch_size=128, gpus=1)
    scores = [score * 100 for score in model_output["scores"]]
    
    return scores


def calculate_token_len(df, tokenizer, col):
    if col != 'src' and col != 'tgt' and not col.endswith('trans'):
        raise ValueError("Column name should be either 'src' or 'tgt', or ends with 'trans'")
    token_lens = [len(tokenizer.tokenize(text)) for text in df[col]]
    len_col = col.replace('trans', 'len') if 'trans' in col else col + 'len'
    df.insert(df.columns.get_loc(col) + 1, len_col, token_lens)
    print(f"Token length of column '{col}' calculated.")
    return df


def calculate_speed(eval_df, tgt_col, len_col='src_len'):
    tqdm_iterator = tqdm(eval_df.iterrows(), total=len(eval_df), desc="Calculating speed")
    token_per_sec = []
    for _, example in tqdm_iterator:
        token_len = int(example[len_col])
        time = float(example[tgt_col])
        token_per_sec.append(token_len / time)
    
    mean_token_per_sec = sum(token_per_sec) / len(token_per_sec) * 1000

    return mean_token_per_sec


def calculate_sacrebleu(eval_df, column_name, ref_col='ko'):
    metric = evaluate.load('sacrebleu')

    references = eval_df[ref_col].tolist()
    references = [[ref] for ref in references]
    candidates = eval_df[column_name].fillna(' ').tolist()

    sacrebleu = metric.compute(references=references, predictions=candidates)['score']

    return sacrebleu


def calculate_sacrebleu_line_by_line(eval_df, column_name, ref_col='ko'):
    metric = evaluate.load('sacrebleu')

    references = eval_df[ref_col].tolist()
    candidates = eval_df[column_name].fillna(' ').tolist()

    sacrebleu_scores = []
    for ref, can in zip(references, candidates):
        sacrebleu = metric.compute(references=[ref], predictions=[can])['score']
        sacrebleu_scores.append(sacrebleu)

    return sacrebleu_scores


def make_eval_dict(results, direction_cols, data_source_cols, save_yaml_path, metric_type='xcomet-no-ref', print_dict=False):
    if 'xcomet' in metric_type:
        xcomet_model = load_xcomet_model()

    eval_dict = read_eval_dict_yaml(save_yaml_path)
    for direction in direction_cols:
        source_dict = dict() if direction not in eval_dict.keys() else eval_dict[direction]
        for data_source in data_source_cols:
            score_dict = dict() if data_source not in source_dict.keys() else source_dict[data_source]
            for column in results.columns:
                if metric_type == 'speed' and 'time' not in column:
                    continue
                if metric_type != 'speed' and 'trans' not in column:
                    continue
                if direction in eval_dict.keys() and data_source in eval_dict[direction].keys() and column in eval_dict[direction][data_source].keys():
                    continue

                target_data = results[(results['data_source'].str.startswith(data_source)) & (results['direction'] == direction)]
                
                print(f"Calculating {metric_type.upper()} of {direction.upper()}-{data_source.upper()}: {column}...")
                if 'xcomet' in metric_type:
                    if metric_type == 'xcomet-no-ref':
                        ref_col = None
                    elif metric_type == 'xcomet-with-ref':
                        ref_col = 'tgt'
                    else:
                        raise ValueError(f"Invalid metric type: {metric_type}")
                    score_dict[column] = calculate_xcomet(target_data, xcomet_model, column, 'src', ref_col)
                elif metric_type == 'speed':
                    score_dict[column] = calculate_speed(target_data, column, column.replace('time', 'len'))
                elif metric_type == 'bleu':
                    score_dict[column] = calculate_sacrebleu(target_data, column, 'tgt')
                else:
                    raise ValueError(f"Invalid metric type: {metric_type}")

            if print_dict:
                print(score_dict)
            
            source_dict[data_source] = score_dict
        eval_dict[direction] = source_dict

    if print_dict:
        print(eval_dict)

    save_eval_dict_yaml(eval_dict, save_yaml_path)

    return eval_dict


def save_line_by_line_metrics(eval_df, save_path, metric_type='xcomet-with-ref', src_col='src', ref_col='tgt'):
    if 'xcomet' in metric_type:
        xcomet_model = load_xcomet_model()

    for tgt_col in eval_df.columns:
        if tgt_col == 'src' or tgt_col == 'tgt':
            if metric_type != 'xcomet-no-ref':
                continue
            if tgt_col == 'src':
                src_col = 'tgt'
                ref_col = 'src'
        else:
            if 'trans' not in tgt_col:
                continue
        
        print(f"Calculating {metric_type.upper()} of {tgt_col}...")

        if 'xcomet' in metric_type:
            if not tgt_col.endswith('trans'):
                eval_col = tgt_col + f'_{metric_type}'
            else:
                eval_col = tgt_col.replace('trans', metric_type)
            if eval_col in eval_df.columns:
                continue
            xcomet_scores = calculate_xcomet_line_by_line(eval_df, xcomet_model, tgt_col, src_col, ref_col)
            if eval_col in eval_df.columns:
                eval_df[eval_col] = xcomet_scores
            else:
                eval_df.insert(eval_df.columns.get_loc(tgt_col) + 1, eval_col, xcomet_scores)

        elif metric_type == 'bleu':
            eval_col = tgt_col.replace('trans', 'bleu')
            if eval_col in eval_df.columns:
                continue
            bleu_scores = calculate_sacrebleu_line_by_line(eval_df, tgt_col, ref_col)
            eval_df.insert(eval_df.columns.get_loc(tgt_col) + 1, eval_col, bleu_scores)

    eval_df.to_csv(save_path, index=False)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_type', type=str, default='api', help='api, llama2, llama3, etc.')
    parser.add_argument('--eval_type', type=str, default='eval-dict', help='eval-dict, line-by-line, all')
    parser.add_argument('--metric_type', type=str, default='bleu', help='xcomet-with-ref, xcomet-no-ref, bleu, speed')
    args = parser.parse_args()

    results_path_dict = {
        'api': '../results/sparta/test_sparta_bidir_api_inferenced.csv',
        'llama2': '../results/sparta/test_sparta_bidir_llama2_inferenced.csv',
        'llama3': '../results/sparta/test_sparta_bidir_llama3_inferenced.csv',
        'ja-base-train': '../results/mmt/ja_base_train.csv',
        'zh-base-train': '../results/mmt/zh_base_train.csv',
        'ja': '../results/mmt/ja_test_bidir_inferenced.csv',
        'zh': '../results/mmt/zh_test_bidir_inferenced.csv',
        'mmt': '../results/mmt/mmt_test_bidir_inferenced.csv', 
        'mmt-m2m': '../results/mmt/mmt_m2m_test_bidir_inferenced.csv',
        'mmt-clean': '../results/mmt/mmt_test_bidir_inferenced_clean.csv',
        'mmt-train-prime': '../results/prime-cleansing/sft/prime_train.csv',
        'mmt-train-base': '../results/prime-cleansing/sft/base_train.csv',
        'en-train-not-prime': '../results/prime-cleansing/sft/en_not_prime_train_with_xcomet.csv',
        'ja-train-not-prime': '../results/prime-cleansing/sft/ja_not_prime_train_with_xcomet.csv',
        'zh-train-not-prime': '../results/prime-cleansing/sft/zh_not_prime_train_with_xcomet.csv',
    }
    results_path = results_path_dict[args.results_type]
    results = pd.read_csv(results_path)
    results.fillna(' ', inplace=True)

    if args.metric_type == 'speed':
        for col in results.columns:
            if col.endswith('trans') or col == 'src' or col == 'tgt':
                if (col != 'src' and col != 'tgt') and (col.replace('trans', 'len') in results.columns):
                    continue
                if (col == 'src' or col == 'tgt') and (col + '_len' in results.columns):
                    continue
                if 'gemma' in col:
                    tokenizer_name = 'google/gemma-7b'
                else:
                    tokenizer_name = 'meta-llama/Meta-Llama-3-8B'
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                results = calculate_token_len(results, tokenizer, col)
        
        results = rearrange_columns(results, front_cols=['tgt', 'tgt_len'], back_cols=['src', 'src_len', 'direction', 'data_source'])
        results.to_csv(results_path, index=False)

    direction_cols = results['direction'].unique()
    data_source_cols = [
        'aihub', 
        'flores'
    ]
    if args.eval_type == 'eval-dict':
        save_path = results_path.replace('inferenced', f'{args.metric_type}').replace('.csv', '.yaml')
        make_eval_dict(results, direction_cols, data_source_cols, save_path, metric_type=args.metric_type, print_dict=True)

    elif args.eval_type == 'line-by-line':
        save_line_by_line_metrics(results, save_path=results_path, metric_type=args.metric_type, src_col='tgt', ref_col='src')

    elif args.eval_type == 'all':
        save_path = results_path.replace('inferenced', f'{args.metric_type}').replace('.csv', '.yaml')
        make_eval_dict(results, direction_cols, data_source_cols, save_path, metric_type=args.metric_type, print_dict=True)
        save_line_by_line_metrics(results, save_path=results_path, metric_type=args.metric_type)


if __name__ == '__main__':
    main()
