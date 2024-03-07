"""
Evaluation metrics for translation quality.

The following functions are available:

[For sentence-level evaluation]
- calculate_sentence_bleu: Calculate the BLEU score for a candidate sentence compared to a reference sentence.
- calculate_sentence_token_bleu: Calculate the BLEU score between a reference sentence and a candidate sentence using tokenization.
- calculate_sentence_sacrebleu: Calculate the SacreBLEU score for a single sentence.
- calculate_sentence_rouge: Calculate the ROUGE scores for a candidate sentence compared to a reference sentence.
- calculate_sentence_wer: Calculate the Word Error Rate (WER) between a reference sentence and a candidate sentence.
- calculate_sentence_bertscore: Calculate the BERTScore for a given reference and candidate sentence.

[For corpus-level evaluation]
- calculate_bleu: Calculate the BLEU score for evaluating translation quality.
- calculate_token_bleu: Calculate the token-level BLEU score for evaluating translation quality.
- calculate_sacrebleu: Calculate the SacreBLEU score for evaluating translation quality.
- calculate_rouge: Calculate the ROUGE-1 and ROUGE-2 scores for the given evaluation dataframe and column name.
- calculate_wer: Calculate the Word Error Rate (WER) for a given evaluation dataframe and column name.
- calculate_bertscore: Calculate the BERTScore for a given evaluation dataframe and column name.

[For evaluation]
- evaluate_all: Evaluate the performance of translation models on multiple columns using multiple metrics.
- evaluate_by_source: Evaluate the performance of the model by source.
- print_evaluation_results: Prints the evaluation results.
- save_eval_results_as_yaml: Save evaluation results as YAML file.
- load_yaml_for_eval_results: Load a YAML file and return the contents as a dictionary.

Examples:
    python inference_evaluation.py --dataset aihub --save_yaml True
    python inference_evaluation.py --dataset flores --save_yaml True

Notes:
    - The evaluation results are saved as a YAML file.
    - The evaluation results are printed to the console.
"""
# built-in
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import yaml

# third-party
import evaluate
import bert_score
import Levenshtein
import pandas as pd
from tqdm import tqdm
from rouge import Rouge
from transformers import AutoTokenizer
from comet import download_model, load_from_checkpoint
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu


def get_average(tuple_data):
    """
    Calculate average of tuple data

    Args:
    - tuple_data (tuple): tuple data (e.g. (1, 2, 3)

    Returns:
    - float: average of tuple data
    """
    return sum(tuple_data) / len(tuple_data)


def calculate_sentence_bleu(reference, candidate):
    """
    Calculate the BLEU score for a candidate sentence compared to a reference sentence.

    Args:
    - reference (str): The reference sentence.
    - candidate (str): The candidate sentence.

    Returns:
    - float: The BLEU score as a percentage.
    """
    bleu = sentence_bleu([reference.split()], candidate.split())
    return bleu * 100


def calculate_bleu(eval_df, column_name):
    """
    Calculate the BLEU score for evaluating translation quality.

    Args:
    - eval_df (pandas.DataFrame): DataFrame containing the evaluation data.
    - column_name (str): Name of the column containing the translated sentences.

    Returns:
    - float: BLEU score as a percentage.
    """
    references = eval_df['ko'].apply(lambda x: [x]).tolist()
    eval_df[column_name] = eval_df[column_name].fillna(' ')
    candidates = eval_df[column_name].tolist()
    bleu = corpus_bleu(references, candidates)
    return bleu * 100


def calculate_sentence_rouge(reference, candidate):
    """
    Calculate the ROUGE scores for a candidate sentence compared to a reference sentence.

    Parameters:
    - reference (str): The reference sentence.
    - candidate (str): The candidate sentence.

    Returns:
    - tuple: A tuple containing the ROUGE-1 score and ROUGE-2 score, both multiplied by 100.
    """
    rouge = Rouge()
    rouge_scores = rouge.get_scores(candidate, reference)
    rouge_1 = rouge_scores[0]['rouge-1']['f']
    rouge_2 = rouge_scores[0]['rouge-2']['f']
    return rouge_1 * 100, rouge_2 * 100


def calculate_rouge(eval_df, column_name):
    """
    Calculate the ROUGE-1 and ROUGE-2 scores for the given evaluation dataframe and column name.

    Parameters:
    - eval_df (pandas.DataFrame): The evaluation dataframe.
    - column_name (str): The name of the column containing the reference sentences.

    Returns:
    - tuple: A tuple containing the ROUGE-1 and ROUGE-2 scores.
    """
    eval_df[column_name] = eval_df[column_name].fillna(' ')
    rouge_1_scores, rouge_2_scores = zip(*eval_df.apply(lambda row: calculate_sentence_rouge(row['ko'], row[column_name]), axis=1))
    rouge_1, rouge_2 = get_average(rouge_1_scores), get_average(rouge_2_scores)
    return rouge_1, rouge_2


def calculate_sentence_wer(reference, candidate):
    """
    Calculates the Word Error Rate (WER) between a reference sentence and a candidate sentence.

    Parameters:
    - reference (str): The reference sentence.
    - candidate (str): The candidate sentence.

    Returns:
    - int: The Word Error Rate (WER) between the reference and candidate sentences.
    """
    return Levenshtein.distance(reference.split(), candidate.split())


def calculate_wer(eval_df, column_name):
    """
    Calculate the Word Error Rate (WER) for a given evaluation dataframe and column name.

    Parameters:
    - eval_df (pandas.DataFrame): The evaluation dataframe containing the reference and hypothesis sentences.
    - column_name (str): The name of the column in the dataframe that contains the hypothesis sentences.

    Returns:
    - float: The calculated Word Error Rate (WER).
    """
    eval_df = eval_df.fillna(' ')
    wer_scores = eval_df.apply(lambda row: calculate_sentence_wer(row['ko'], row[column_name]), axis=1)
    wer = get_average(wer_scores)
    return wer


def calculate_sentence_token_bleu(reference, candidate, tokenizer_name='gogamza/kobart-base-v2'):
    """
    Calculates the BLEU score between a reference sentence and a candidate sentence.

    Args:
    - reference (str): The reference sentence.
    - candidate (str): The candidate sentence.
    - tokenizer_name (str, optional): The name of the tokenizer to use. Defaults to 'gogamza/kobart-base-v2'.

    Returns:
    - float: The BLEU score between the reference and candidate sentences.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    reference_tokens = tokenizer.tokenize(reference)
    candidate_tokens = tokenizer.tokenize(candidate)

    token_bleu = sentence_bleu([reference_tokens], candidate_tokens) * 100
    
    return token_bleu


def calculate_token_bleu(eval_df, column_name, tokenizer_name='gogamza/kobart-base-v2'):
    """
    Calculate the token-level BLEU score for evaluating translation quality.

    Args:
    - eval_df (pandas.DataFrame): DataFrame containing the evaluation data.
    - column_name (str): Name of the column containing the translations to be evaluated.
    - tokenizer_name (str, optional): Name of the tokenizer to be used. Defaults to 'gogamza/kobart-base-v2'.

    Returns:
    - float: The token-level BLEU score.

    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    references = eval_df['ko'].tolist()
    candidates = eval_df[column_name].fillna(' ').tolist()

    references = [tokenizer.tokenize(ref) for ref in references]
    candidates = [tokenizer.tokenize(can) for can in candidates]

    token_bleu_scores = [sentence_bleu(references, candidate) for candidate in tqdm(candidates)]
    token_bleu = get_average(token_bleu_scores) * 100

    return token_bleu


def calculate_sentence_sacrebleu(reference, candidate):
    """
    Calculate the SacreBLEU score for a single sentence.

    Args:
    - reference (str): The reference sentence.
    - candidate (str): The candidate sentence.

    Returns:
    - float: The SacreBLEU score.
    """
    metric = evaluate.load('sacrebleu')
    candidate = candidate if not pd.isna(candidate) else ' '
    sacrebleu = metric.compute(references=[[reference]], predictions=[candidate])['score']
    return sacrebleu


def calculate_sacrebleu(eval_df, column_name):
    """
    Calculate the SacreBLEU score for evaluating translation quality.

    Args:
    - eval_df (pandas.DataFrame): DataFrame containing the evaluation data.
    - column_name (str): Name of the column containing the translations to be evaluated.

    Returns:
    - float: The SacreBLEU score.

    """
    metric = evaluate.load('sacrebleu')

    references = eval_df['ko'].tolist()
    references = [[ref] for ref in references]
    candidates = eval_df[column_name].fillna(' ').tolist()

    sacrebleu = metric.compute(references=references, predictions=candidates)['score']

    return sacrebleu


def calculate_sentence_bertscore(reference, candidate):
    """
    Calculate the BERTScore for a given reference and candidate sentence.

    Parameters:
    - reference (str): The reference sentence.
    - candidate (str): The candidate sentence.

    Returns:
    - float: The BERTScore value for the candidate sentence.
    """
    reference = [reference]
    candidate = candidate if candidate else ' '
    candidate = [candidate]

    _, _, bertscore = bert_score.score(reference, candidate, lang='ko')

    return bertscore.item() * 100


def calculate_bertscore(eval_df, column_name):
    """
    Calculate the BERTScore for a given evaluation dataframe and column name.

    Parameters:
    - eval_df (pandas.DataFrame): The evaluation dataframe containing the reference and candidate sentences.
    - column_name (str): The name of the column in the dataframe containing the candidate sentences.

    Returns:
    - float: The average BERTScore multiplied by 100.
    """
    references = eval_df['ko'].tolist()
    candidates = eval_df[column_name].fillna(' ').tolist()

    _, _, bertscore = bert_score.score(references, candidates, lang='ko')

    avg_bertscore = bertscore.mean().item()

    return avg_bertscore * 100


def calculate_sentence_xcomet(source, reference, candidate):
    model_path = download_model("Unbabel/XCOMET-XL")
    model = load_from_checkpoint(model_path)
    triplets = [{"src": source, "mt": candidate, "ref": reference}]
    model_output = model.predict(triplets, batch_size=1, gpus=1)
    xcomet = model_output[0][0]
    return xcomet * 100


def calculate_xcomet(eval_df, column_name):
    model_path = download_model("Unbabel/XCOMET-XL")
    model = load_from_checkpoint(model_path)
    triplets = []
    for _, example in eval_df.iterrows():
        src_text = example['en']
        mt_text = example[column_name] if not pd.isna(example[column_name]) else ' '
        ref_text = example['ko']

        triplet = {"src": src_text, "mt": mt_text, "ref": ref_text}
        triplets.append(triplet)
    model_output = model.predict(triplets, batch_size=16, gpus=1)
    xcomet = model_output[1]
    return xcomet * 100


def evaluate_all(eval_df, column_list=None, metric_list=None):
    """
    Evaluate the performance of translation models on multiple columns using multiple metrics.

    Args:
    - eval_df (pandas.DataFrame): The DataFrame containing the evaluation data.
    - column_list (list, optional): The list of columns to evaluate. Defaults to ['google_trans'].
    - metric_list (list, optional): The list of metrics to calculate. Defaults to ['sacrebleu'].

    Returns:
    - eval_dict (dict): A dictionary containing the evaluation results for each column.
    """
    if column_list is None:
        column_list = ['google_trans']
    if metric_list is None:
        metric_list = ['sacrebleu']

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
        if 'comet' in metric_list:
            mean_comet = calculate_xcomet(eval_df, col)
            col_dict['comet'] = mean_comet

        eval_dict[col] = col_dict

    return eval_dict


def evaluate_by_source(eval_df, source_list, column_list, metric_list):
    """
    Evaluate the performance of the model by source.

    Args:
    - eval_df (pandas.DataFrame): The evaluation dataframe.
    - source_list (list): List of sources to evaluate.
    - column_list (list): List of columns to evaluate.
    - metric_list (list): List of metrics to evaluate.

    Returns:
    - eval_dict_by_source (dict): A dictionary containing the evaluation results for each source.
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
    Prints the evaluation results.

    Args:
    - eval_dict (dict): A dictionary containing the evaluation results.
    """
    for trans in eval_dict.keys():
        print(f"[{str(trans).upper()}]")
        for metric_key, metric_value in eval_dict[trans].items():
            if isinstance(metric_value, dict):
                print(f" - {str(metric_key).upper()}")
                for metric, value in metric_value.items():
                    print(f" - {metric}: {value}")
            else:
                print(f" - {str(metric_key).upper()}: {metric_value}")


def save_eval_results_as_yaml(eval_dict, save_path):
    """
    Save evaluation results as YAML file.

    Args:
    - eval_dict (dict): Dictionary containing evaluation results.
    - save_path (str): Path to save the YAML file.
    """
    with open(save_path, 'w') as file:
        yaml.dump(eval_dict, file)


def load_yaml_for_eval_results(yaml_path):
    """
    Load a YAML file and return the contents as a dictionary.

    Parameters:
    - yaml_path (str): The path to the YAML file.

    Returns:
    - eval_dict (dict): The contents of the YAML file as a dictionary.
    """
    with open(yaml_path, 'r') as file:
        eval_dict = yaml.safe_load(file)
    return eval_dict


if __name__ == '__main__':
    import argparse
    """
    [COLUMN_LIST]
    # API
    - papago_trans: Naver Papago API
    - google_trans: Google Translate API
    - deepl_trans: DeepL API

    # Model Checkpoints
    - mbart_trans: facebook/mbart-large-50-many-to-many-mmt
    - nllb-600m_trans: facebook/nllb-200-distilled-600M
    - nllb-1.3b_trans: facebook/nllb-200-distilled-1.3B
    - madlad_trans: google/madlad400-3b-mt

    # Finetuned 
    -----------------------------------------------------------------------------------------------
      model: facebook/mbart-large-50 (mbart)
             beomi/open-llama-2-ko-7b (llama)
      dataset: traintogpb/aihub-koen-translation-integrated-tiny-100k
    -----------------------------------------------------------------------------------------------
    - mbart-aihub_trans: Full-finetuned on aihub dataset
    - llama_trans: Pretrained only
    - llama-aihub-qlora_trans: QLORA finetuned on aihub dataset
    - llama-aihub-qlora-bf16_trans: Upscaled with BrainFloat 16-bit from QLORA finetuned version
    - llama-aihub-qlora-fp16_trans: Upscaled with Float 16-bit from QLORA finetuned version
    - llama-aihub-qlora-augment_trans: QLoRA finetuned on augmented aihub dataset (240k)
    - llama-aihub-qlora-reverse-new_trans: QLoRA finetuned on reverse direction (ko-en) from llama-aihub-qlora checkpoint with the new data
    - llama-aihub-qlora-reverse-overlap_trans: QLoRA finetuned on reverse direction (ko-en) from llama-aihub-qlora checkpoint with the same data

    [METRIC_LIST]
    bleu: 
    - Bi-Lingual Evaluation Understudy
    - generation 단어가 reference에 얼마나 포함되는지
    - 번역에서 가장 흔하게 사용되나 띄어쓰기 기준 단어 단위로 평가해 한국어 평가에 다소 부적합한 면이 있음
    sacrebleu: 
    - SacreBLEU
    - BLEU와 원리가 같으나 띄어쓰기 단어 기준인 BLEU와 달리 토큰 단위로 평가해 한국어 평가에 조금 더 적합
    - 배포된 국제 표준이 있어 통일된 평가가 가능
    rouge: 
    - Recall-Oriented Understudy for Gisting Evaluation
    - reference 단어가 generation에 얼마나 포함되는지
    - 번역보다는 요약 task에서 주로 사용하며, Rouge-L도 있으나 거의 Rouge-1과 동일하여 제외
    wer: 
    - Word Error Rate
    - 단순하게 generation과 reference 간 단어 단위 오류율로 계산
    - 번역보다는 음성 인식 task에서 주로 사용
    bertscore:
    - BERTScore
    - reference와 generation을 BERT 기반으로 임베딩하여 cosine similarity로 평가
    - 번역의 품질을 평가하는데 가장 최신이자 성능이 좋은 metric

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
        # 'llama-aihub-qlora_trans',
        # 'llama-aihub-qlora-bf16_trans',
        # 'llama-aihub-qlora-fp16_trans',
        'llama-aihub-qlora-bf16-vllm_trans',
        # 'llama-aihub-qlora-augment_trans',
        # 'llama-aihub-qlora-reverse-new_trans',
        # 'llama-aihub-qlora-reverse-overlap_trans',
        'mt5-aihub-base-fft_trans'
    ]
    metric_list = [
        'sacrebleu', 
        'bertscore', 
        'comet'
    ]
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
