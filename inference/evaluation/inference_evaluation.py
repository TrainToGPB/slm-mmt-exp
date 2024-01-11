import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge import Rouge
import Levenshtein
from transformers import AutoTokenizer
import evaluate
from tqdm import tqdm


def get_average(tuple_data):
    return sum(tuple_data) / len(tuple_data)


def calculate_sentence_bleu(reference, candidate):
    bleu = sentence_bleu([list(reference.split())], list(candidate.split()))
    return bleu * 100


def calculate_bleu(eval_df, column_name):
    references = eval_df['ko'].apply(lambda x: [x]).tolist()
    candidates = eval_df[column_name].tolist()
    bleu = corpus_bleu(references, candidates)
    return bleu * 100


def calculate_sentence_rouge(reference, candidate):
    rouge = Rouge()
    rouge_scores = rouge.get_scores(candidate, reference)
    rouge_1 = rouge_scores[0]['rouge-1']['f']
    rouge_2 = rouge_scores[0]['rouge-2']['f']
    return rouge_1 * 100, rouge_2 * 100


def calculate_rouge(eval_df, column_name):
    rouge_1_scores, rouge_2_scores = zip(*eval_df.apply(lambda row: calculate_sentence_rouge(row['ko'], row[column_name]), axis=1))
    rouge_1, rouge_2 = get_average(rouge_1_scores), get_average(rouge_2_scores)
    return rouge_1, rouge_2


def calculate_sentence_wer(reference, candidate):
    return Levenshtein.distance(reference.split(), candidate.split())


def calculate_wer(eval_df, column_name):
    wer_scores = eval_df.apply(lambda row: calculate_sentence_wer(row['ko'], row[column_name]), axis=1)
    wer = get_average(wer_scores)
    return wer


def calculate_sacrebleu(eval_df, column_name, tokenizer_name='gogamza/kobart-base-v2'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    references = eval_df['ko'].tolist()
    candidates = eval_df[column_name].tolist()

    references = [tokenizer.tokenize(ref) for ref in references]
    candidates = [tokenizer.tokenize(can) for can in candidates]

    # sacrebleu = evaluate.load('sacrebleu')
    sacrebleu_scores = [sentence_bleu(references, candidate) for candidate in tqdm(candidates)]
    sacrebleu = get_average(sacrebleu_scores) * 100

    return sacrebleu


def evaluate_all(eval_df, column_list, metric_list):
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

        eval_dict[col] = col_dict

    return eval_dict


def print_evaluation_results(eval_dict):
    for trans in eval_dict.keys():
        print(f"[{trans.upper()}]")
        for metric_key, metric_value in eval_dict[trans].items():
            print(f" - {metric_key.upper()}: {metric_value:.2f}")


if __name__ == '__main__':
    eval_path = '../results/test_tiny_uniform100_inferenced.csv'
    eval_df = pd.read_csv(eval_path)
    
    # evaluate sentence
    # u100_eval = pd.read_csv(eval_path)
    # sentence_num = 200
    # reference = u100_eval['ko'][sentence_num]
    # candidate = u100_eval['deepl_trans'][sentence_num]
    # print(calculate_sentence_bleu(reference, candidate))
    # print(calculate_sentence_rouge(reference, candidate))

    # evaluate dataset
    """
    [COLUMN_LIST]
    papago_trans: 네이버 파파고 API (현재 개수 미달)
    google_trans: 구글 번역 API
    deepl_trans: 딥엘 API
    opus_trans: Helsinki-NLP/opus-mt-tc-big-en-ko (HuggingFace)
    mbart_trans: facebook/mbart-large-50-many-to-many-mmt (HuggingFace)
    nllb_trans: facebook/nllb-200-distilled-600M (HuggingFace)
    """
    column_list = ['nllb_trans']
    metric_list = ['bleu', 'sacrebleu', 'rouge', 'wer']
    eval_dict = evaluate_all(eval_df, column_list, metric_list)
    print_evaluation_results(eval_dict)
