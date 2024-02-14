import yaml

import evaluate
import bert_score
import Levenshtein
import pandas as pd
from tqdm import tqdm
from rouge import Rouge
from transformers import AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu


def get_average(tuple_data):
    """
    Calculate the average of a tuple of numerical values.

    Parameters:
    - tuple_data (tuple): Tuple of numerical values.

    Returns:
    - average (float): Average value.
    """
    return sum(tuple_data) / len(tuple_data)


def calculate_sentence_bleu(reference, candidate):
    """
    Calculate the BLEU score for a pair of reference and candidate sentences.

    Parameters:
    - reference (str): Reference sentence.
    - candidate (str): Candidate sentence.

    Returns:
    - bleu (float): BLEU score.
    """
    bleu = sentence_bleu([reference.split()], candidate.split())
    return bleu * 100


def calculate_bleu(eval_df, column_name):
    """
    Calculate the BLEU score for a given column in the evaluation dataframe.

    Parameters:
    - eval_df (DataFrame): Evaluation dataframe.
    - column_name (str): Name of the column to be evaluated.

    Returns:
    - bleu (float): BLEU score.
    """
    references = eval_df['ko'].apply(lambda x: [x]).tolist()
    eval_df[column_name] = eval_df[column_name].fillna(' ')
    candidates = eval_df[column_name].tolist()
    bleu = corpus_bleu(references, candidates)
    return bleu * 100


def calculate_sentence_rouge(reference, candidate):
    """
    Calculate the ROUGE scores (ROUGE-1 and ROUGE-2) for a pair of reference and candidate sentences.

    Parameters:
    - reference (str): Reference sentence.
    - candidate (str): Candidate sentence.

    Returns:
    - rouge_1 (float): ROUGE-1 score.
    - rouge_2 (float): ROUGE-2 score.
    """
    rouge = Rouge()
    rouge_scores = rouge.get_scores(candidate, reference)
    rouge_1 = rouge_scores[0]['rouge-1']['f']
    rouge_2 = rouge_scores[0]['rouge-2']['f']
    return rouge_1 * 100, rouge_2 * 100


def calculate_rouge(eval_df, column_name):
    """
    Calculate the average ROUGE scores (ROUGE-1 and ROUGE-2) for a given column in the evaluation dataframe.

    Parameters:
    - eval_df (DataFrame): Evaluation dataframe.
    - column_name (str): Name of the column to be evaluated.

    Returns:
    - rouge_1 (float): Average ROUGE-1 score (percentage).
    - rouge_2 (float): Average ROUGE-2 score (percentage).
    """
    eval_df[column_name] = eval_df[column_name].fillna(' ')
    rouge_1_scores, rouge_2_scores = zip(*eval_df.apply(lambda row: calculate_sentence_rouge(row['ko'], row[column_name]), axis=1))
    rouge_1, rouge_2 = get_average(rouge_1_scores), get_average(rouge_2_scores)
    return rouge_1, rouge_2


def calculate_sentence_wer(reference, candidate):
    """
    Calculate the Word Error Rate (WER) for a pair of reference and candidate sentences.

    Parameters:
    - reference (str): Reference sentence.
    - candidate (str): Candidate sentence.

    Returns:
    - wer (int): Word Error Rate.
    """
    return Levenshtein.distance(reference.split(), candidate.split())


def calculate_wer(eval_df, column_name):
    """
    Calculate the average Word Error Rate (WER) for a given column in the evaluation dataframe.

    Parameters:
    - eval_df (DataFrame): Evaluation dataframe.
    - column_name (str): Name of the column to be evaluated.

    Returns:
    - wer (float): Average Word Error Rate.
    """
    eval_df = eval_df.fillna(' ')
    wer_scores = eval_df.apply(lambda row: calculate_sentence_wer(row['ko'], row[column_name]), axis=1)
    wer = get_average(wer_scores)
    return wer


def calculate_sentence_token_bleu(reference, candidate, tokenizer_name='gogamza/kobart-base-v2'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    reference_tokens = tokenizer.tokenize(reference)
    candidate_tokens = tokenizer.tokenize(candidate)

    sacrebleu = sentence_bleu([reference_tokens], candidate_tokens) * 100
    
    return sacrebleu


def calculate_token_bleu(eval_df, column_name, tokenizer_name='gogamza/kobart-base-v2'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    references = eval_df['ko'].tolist()
    candidates = eval_df[column_name].fillna(' ').tolist()

    references = [tokenizer.tokenize(ref) for ref in references]
    candidates = [tokenizer.tokenize(can) for can in candidates]

    # sacrebleu = evaluate.load('sacrebleu')
    sacrebleu_scores = [sentence_bleu(references, candidate) for candidate in tqdm(candidates)]
    sacrebleu = get_average(sacrebleu_scores) * 100

    return sacrebleu


def calculate_sentence_sacrebleu(reference, candidate):
    metric = evaluate.load('sacrebleu')
    candidate = candidate if not pd.isna(candidate) else ' '
    sacrebleu = metric.compute(references=[[reference]], predictions=[candidate])['score']
    return sacrebleu


def calculate_sacrebleu(eval_df, column_name):
    metric = evaluate.load('sacrebleu')

    references = eval_df['ko'].tolist()
    references = [[ref] for ref in references]
    candidates = eval_df[column_name].fillna(' ').tolist()

    sacrebleu = metric.compute(references=references, predictions=candidates)['score']

    return sacrebleu


def calculate_sentence_bertscore(reference, candidate):
    reference = [reference]
    candidate = candidate if candidate else ' '
    candidate = [candidate]

    _, _, bertscore = bert_score.score(reference, candidate, lang='ko')

    return bertscore.item() * 100


def calculate_bertscore(eval_df, column_name):
    references = eval_df['ko'].tolist()
    candidates = eval_df[column_name].fillna(' ').tolist()

    _, _, bertscore = bert_score.score(references, candidates, lang='ko')

    avg_bertscore = bertscore.mean().item()

    return avg_bertscore * 100


def evaluate_all(eval_df, column_list=None, metric_list=None):
    """
    Evaluate multiple metrics for each column in the evaluation dataframe.

    Parameters:
    - eval_df (DataFrame): Evaluation dataframe.
    - column_list (list): List of column names to be evaluated.
    - metric_list (list): List of metrics to be calculated ('bleu', 'sacrebleu', 'rouge', 'wer').

    Returns:
    - eval_dict (dict): Dictionary containing evaluation results for each column and metric.
    """
    if column_list is None:
        column_list = ['google_trans', 'deepl_trans','mbart_trans', 'nllb-600m_trans', 'nllb-1.3b_trans', 'madlad_trans']
    if metric_list is None:
        metric_list = ['bleu', 'bertscore']

    eval_dict = dict()
    for col in column_list:
        print(f"Evaluting Column {col}..")
        col_dict = dict()
        if 'bleu' in metric_list:
            mean_bleu = calculate_bleu(eval_df, col)
            col_dict['bleu'] = mean_bleu
        if 'sacrebleu' in metric_list:
            mean_sacrebleu = calculate_sacrebleu(eval_df, col)
            col_dict['sacrebleu'] = mean_sacrebleu
        if 'rouge' in metric_list:
            mean_rouge_1, mean_rouge_2 = calculate_rouge(eval_df, col)
            col_dict['rouge_1'] = mean_rouge_1
            col_dict['rouge_2'] = mean_rouge_2
        if 'wer' in metric_list:
            mean_wer = calculate_wer(eval_df, col)
            col_dict['wer'] = mean_wer
        if 'bertscore' in metric_list:
            mean_bertscore = calculate_bertscore(eval_df, col)
            col_dict['bertscore'] = mean_bertscore

        eval_dict[col] = col_dict

    return eval_dict


def evaluate_by_source(eval_df, source_list, column_list, metric_list):
    """
    Evaluate multiple metrics for each column in the evaluation dataframe, grouped by source.

    Parameters:
    - eval_df (DataFrame): Evaluation dataframe.
    - source_list (list): List of source names.
    - column_list (list): List of column names to be evaluated.
    - metric_list (list): List of metrics to be calculated ('bleu', 'sacrebleu', 'rouge', 'wer', 'bertscore').

    Returns:
    - eval_dict_by_source (dict): Dictionary containing evaluation results for each source, column, and metric.
    """
    eval_dict_by_source = dict()
    for src in source_list:
        print(f'\n[EVALUATING SOURCE: {src}]')
        subset_eval_df = eval_df[eval_df['source'] == src]
        subset_eval_dict = evaluate_all(subset_eval_df, column_list, metric_list)
        eval_dict_by_source[src] = subset_eval_dict
    return eval_dict_by_source


def print_evaluation_results(eval_dict):
    """
    Print the evaluation results for each translation model and metric.

    Parameters:
    - eval_dict (dict): Dictionary containing evaluation results.

    Output:
    - Printed results in the console.
    """
    for trans in eval_dict.keys():
        print(f"[{str(trans).upper()}]")
        for metric_key, metric_value in eval_dict[trans].items():
            if isinstance(metric_value, dict):
                print(f" - {str(metric_key).upper()}")
                for metric, value in metric_value.items():
                    print(f" - {metric}: {value:.2f}")
            else:
                print(f" - {str(metric_key).upper()}: {metric_value:.2f}")


def save_eval_results_as_yaml(eval_dict, save_path):
    """
    Save the evaluation results to a YAML file.

    Parameters:
    - eval_dict (dict): Dictionary containing evaluation results.
    - file_path (str): Path to the YAML file where results will be saved.
    """
    with open(save_path, 'w') as file:
        yaml.dump(eval_dict, file)


def load_yaml_for_eval_results(yaml_path):
    with open(yaml_path, 'r') as file:
        eval_dict = yaml.safe_load(file)
    return eval_dict


if __name__ == '__main__':
    import argparse
    """
    [COLUMN_LIST]
    papago_trans: 네이버 파파고 API (현재 개수 미달)
    google_trans: 구글 번역 API
    deepl_trans: 딥엘 API
    mbart_trans: facebook/mbart-large-50-many-to-many-mmt (HuggingFace)
    mbart-aihub_trans: facebook/mbart-large-50 (HuggingFace) + AI Hub 한-영 번역 데이터 full-finetuning
    nllb-600m_trans: facebook/nllb-200-distilled-600M (HuggingFace)
    nllb-1.3b_trans: facebook/nllb-200-distilled-1.3B (HuggingFace)
    madlad_trans: google/madlad400-3b-mt (HuggingFace)
    llama_trans: beomi/open-llama-ko-7b (HuggingFace)
    llama-aihub-qlora_trans: beomi/open-llama-ko-7b (HuggingFace) + AI Hub 한-영 번역 데이터 QLoRA finetuning
    llama-aihub-qlora_trans_processed: llama-aihub-qlora_trans의 수작업 정제 버전

    [METRIC_LIST]
    bleu: 
     - BLEU (Bi-Lingual Evaluation Understudy)
     - 번역에서 가장 흔하게 사용되나 띄어쓰기 기준 단어 단위로 평가해 한국어 평가에 다소 부적합한 면이 있음
     - generation 단어가 reference에 얼마나 포함되는지
    sacrebleu: 
     - SacreBLEU
     - BLEU와 원리가 같으나 띄어쓰기 단어 기준인 BLEU와 달리 토큰 단위로 평가해 한국어 평가에 조금 더 적합
    rouge: 
     - Rouge-1 / Rouge-2
     - 번역보다는 요약 task에서 주로 사용하며, Rouge-L도 있으나 거의 Rouge-1과 동일하여 제외
     - reference 단어가 generation에 얼마나 포함되는지
    wer: 
     - Word Error Rate
     - 번역보다는 음성 인식 task에서 주로 사용
     - 단순하게 generation과 reference 간 단어 단위 오류율로 계산

    [SOURCE_LIST]
    - 111: 전문분야
    - 124: 기술과학1
    - 125: 사회과학
    - 126: 일반
    - 563: 산업정보(특허)
    - 71265: 일상생활 및 구어체
    - 71266: 기술과학2
    - 71382: 방송콘텐츠
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='aihub', help="Dataset for evaluation")
    parser.add_argument("--save_yaml", type=lambda x: (str(x).lower() == 'true'), default=False, help="Save (or not) evaluation results in yaml file")
    args = parser.parse_args()
    dataset = args.dataset
    save_yaml = args.save_yaml

    if dataset == 'aihub':
        eval_path = '../results/test_tiny_uniform100_inferenced.csv'
        save_path = '../results/test_tiny_uniform100_metrics.yaml'
        save_path_by_source = '../results/test_tiny_uniform100_metrics_by_source.yaml'
    elif dataset == 'flores':
        eval_path = '../results/test_flores_inferenced.csv'
        save_path = '../results/test_flores_metrics.yaml'

    eval_df = pd.read_csv(eval_path)

    column_list = [
        'papago_trans',
        'google_trans', 
        'deepl_trans', 
        'mbart_trans', 
        'nllb-600m_trans', 
        'madlad_trans', 
        'mbart-aihub_trans', 
        'llama-aihub-qlora_trans',
        'llama-aihub-qlora-bf16_trans',
        'llama-aihub-qlora-fp16_trans',
        'llama-aihub-qlora-augment_trans',
        'llama-aihub-qlora-reverse-new_trans',
        'llama-aihub-qlora-reverse-overlap_trans',
    ]
    metric_list = ['sacrebleu', 'bertscore']
    source_list = [
        111, 
        124, 
        125, 
        126, 
        563, 
        71265, 
        71266, 
        71382
    ]

    # evaluate all
    eval_dict = evaluate_all(eval_df, column_list, metric_list)
    print_evaluation_results(eval_dict)
    if save_yaml:
        save_eval_results_as_yaml(eval_dict, save_path)

    # evaluate separately by source (only for aihub dataset)
    if dataset == 'aihub':
        eval_dict_by_source = evaluate_by_source(eval_df, source_list, column_list, metric_list)
        print_evaluation_results(eval_dict_by_source)
        if save_yaml:
            save_eval_results_as_yaml(eval_dict_by_source, save_path_by_source)

    # SacreBLEU는 7.18/100점인데 반해, BERTScore는 89.33/100점
    # reference = "미국 심장협회의 연구에 따르면 이 행동을 자주 하면 고혈압을 의심해 봐야 한다고 하는데요."
    # candidate = "美심장협회 연구에 따르면, 이렇게 자주 한다면 고혈압을 의심해야 한다."
    # score = calculate_sentence_bertscore(reference, candidate)
    # print(score)
