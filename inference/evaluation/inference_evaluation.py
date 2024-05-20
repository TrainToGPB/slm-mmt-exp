import yaml
from tqdm import tqdm

import evaluate
import pandas as pd
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
        ref_text = example[ref_col]

        triplet = {"src": src_text, "mt": tgt_text, "ref": ref_text}
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
    scores = model_output["scores"]
    
    return scores


def calculate_speed(eval_df, tgt_col, len_col='src_token_len'):
    tqdm_iterator = tqdm(eval_df.iterrows(), total=len(eval_df), desc="Calculating speed")
    token_per_sec = []
    for _, example in tqdm_iterator:
        src_len = example[len_col]
        time = example[tgt_col]

        token_per_sec.append(src_len / time)

    mean_token_per_sec = sum(token_per_sec) / len(token_per_sec) * 1000

    return mean_token_per_sec


def calculate_sacrebleu(eval_df, column_name, ref_col='ko'):
    metric = evaluate.load('sacrebleu')

    references = eval_df[ref_col].tolist()
    references = [[ref] for ref in references]
    candidates = eval_df[column_name].fillna(' ').tolist()

    sacrebleu = metric.compute(references=references, predictions=candidates)['score']

    return sacrebleu


def calculate_sacrebleu(eval_df, column_name, ref_col='ko'):
    metric = evaluate.load('sacrebleu')

    references = eval_df[ref_col].tolist()
    candidates = eval_df[column_name].fillna(' ').tolist()

    sacrebleu_scores = []
    for ref, can in zip(references, candidates):
        sacrebleu = metric.compute(references=[ref], predictions=[can])['score']
        sacrebleu_scores.append(sacrebleu)

    return sacrebleu_scores


def make_eval_dict(results, direction_cols, data_source_cols, save_yaml_path, metric_type='xcomet', print_dict=False):
    if metric_type == 'xcomet':
        xcomet_model = load_xcomet_model()

    eval_dict = read_eval_dict_yaml(save_yaml_path)
    for direction in direction_cols:
        src_lang = direction.split('2')[0]
        source_dict = dict()
        for data_source in data_source_cols:
            score_dict = dict()
            for column in results.columns:
                if 'trans' not in column or 'llama-3' not in column:
                    continue
                if direction in eval_dict and data_source in eval_dict[direction] and column in eval_dict[direction][data_source]:
                    continue

                target_data = results[results['data_source'].str.startswith(data_source) & results['direction'].str.startswith(src_lang)]
                
                print(f"Calculating {metric_type.upper()} of {direction.upper()}-{data_source.upper()}: {column}...")
                if metric_type == 'xcomet':
                    score_dict[column] = calculate_xcomet(target_data, xcomet_model, column, 'src', 'tgt')
                elif metric_type == 'speed':
                    score_dict[column] = calculate_speed(target_data, column, 'src_token_len')
                elif metric_type == 'bleu':
                    score_dict[column] = calculate_sacrebleu(target_data, column, 'tgt')

            print(score_dict)
                
            source_dict[data_source] = score_dict
        eval_dict[direction] = source_dict

    if print_dict:
        print(eval_dict)

    save_eval_dict_yaml(eval_dict, save_yaml_path)

    return eval_dict


def save_line_by_line_metrics(eval_df, save_path, metric_type='xcomet', src_col='src', ref_col='tgt'):
    if metric_type == 'xcomet':
        xcomet_model = load_xcomet_model()

    for tgt_col in eval_df.columns:
        # if 'trans' not in tgt_col:
        #     continue
        if tgt_col != 'ko':
            continue
        
        print(f"Calculating {metric_type.upper()} of {tgt_col}...")

        if metric_type == 'xcomet':
            # eval_col = tgt_col.replace('trans', 'xcomet')
            eval_col = tgt_col + '_xcomet'
            if eval_col in eval_df.columns:
                continue
            # xcomet_scores = calculate_xcomet_line_by_line(eval_df, xcomet_model, tgt_col, src_col, ref_col)
            xcomet_scores = calculate_xcomet_line_by_line(eval_df, xcomet_model, 'ko', src_col)
            eval_df.insert(eval_df.columns.get_loc(tgt_col) + 1, eval_col, xcomet_scores)

        elif metric_type == 'bleu':
            eval_col = tgt_col.replace('trans', 'bleu')
            if eval_col in eval_df.columns:
                continue
            bleu_scores = calculate_sacrebleu(eval_df, tgt_col, ref_col)
            eval_df.insert(eval_df.columns.get_loc(tgt_col) + 1, eval_col, bleu_scores)

    eval_df.to_csv(save_path, index=False)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_type', type=str, default='api', help='api, llama2, llama3')
    parser.add_argument('--inference_type', type=str, default='eval_dict', help='eval_dict, line_by_line, all')
    parser.add_argument('--metric_type', type=str, default='bleu', help='xcomet, bleu, speed')
    args = parser.parse_args()

    results_path_dict = {
        'api': '../results/test_sparta_bidir_api_inferenced.csv',
        'llama2': '../results/test_sparta_bidir_llama2_inferenced.csv',
        'llama3': '../results/test_sparta_bidir_llama3_inferenced.csv',
        'mini-1m-train': '../results/mini-train.csv'
    }
    results_path = results_path_dict[args.results_type]
    results = pd.read_csv(results_path)

    if args.inference_type == 'eval_dict':
        direction_cols = ['en2ko', 'ko2en']
        data_source_cols = ['aihub', 'flores']

        save_path = results_path_dict.replace('inferenced', f'{args.metric_type}').replace('.csv', '.yaml')
        make_eval_dict(results, direction_cols, data_source_cols, save_path, metric_type=args.metric_type, print_dict=True)

    elif args.inference_type == 'line_by_line':
        save_line_by_line_metrics(results, save_path=results_path, metric_type=args.metric_type, src_col='en', ref_col=None)

    elif args.inference_type == 'all':
        direction_cols = ['en2ko', 'ko2en']
        data_source_cols = ['aihub', 'flores']

        save_path = results_path_dict.replace('inferenced', f'{args.metric_type}').replace('.csv', '.yaml')
        make_eval_dict(results, direction_cols, data_source_cols, save_path, metric_type=args.metric_type, print_dict=True)

        save_line_by_line_metrics(results, save_path=results_path, metric_type=args.metric_type)


if __name__ == '__main__':
    main()
