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


def calculate_bleu(eval_path):
    eval_df = pd.read_csv(eval_path)

    references = eval_df['ko'].apply(lambda x: [x]).tolist()
    candidates_google = eval_df['google_trans'].tolist()
    candidates_deepl = eval_df['deepl_trans'].tolist()

    bleu_google = corpus_bleu(references, candidates_google)
    bleu_deepl = corpus_bleu(references, candidates_deepl)

    return bleu_google * 100, bleu_deepl * 100


def calculate_sentence_rouge(reference, candidate):
    rouge = Rouge()
    rouge_scores = rouge.get_scores(candidate, reference)
    rouge_1 = rouge_scores[0]['rouge-1']['f']
    rouge_2 = rouge_scores[0]['rouge-2']['f']
    return rouge_1 * 100, rouge_2 * 100


def calculate_rouge(eval_path):
    eval_df = pd.read_csv(eval_path)
    rouge_1_google, rouge_2_google = zip(*eval_df.apply(lambda row: calculate_sentence_rouge(row['ko'], row['google_trans']), axis=1))
    rouge_1_deepl, rouge_2_deepl = zip(*eval_df.apply(lambda row: calculate_sentence_rouge(row['ko'], row['deepl_trans']), axis=1))
    return (get_average(rouge_1_google), get_average(rouge_2_google)), (get_average(rouge_1_deepl), get_average(rouge_2_deepl))


def calculate_sentence_wer(reference, candidate):
    return Levenshtein.distance(reference.split(), candidate.split())


def calculate_wer(eval_path):
    eval_df = pd.read_csv(eval_path)
    wer_google = eval_df.apply(lambda row: calculate_sentence_wer(row['ko'], row['google_trans']), axis=1)
    wer_deepl = eval_df.apply(lambda row: calculate_sentence_wer(row['ko'], row['deepl_trans']), axis=1)
    return get_average(wer_google), get_average(wer_deepl)


def calculate_sacrebleu(eval_path, tokenizer_name='gogamza/kobart-base-v2'):
    eval_df = pd.read_csv(eval_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    references = eval_df['ko'].tolist()
    candidates_google = eval_df['google_trans'].tolist()
    candidates_deepl = eval_df['deepl_trans'].tolist()

    references = [tokenizer.tokenize(ref) for ref in references]
    candidates_google = [tokenizer.tokenize(can) for can in candidates_google]
    candidates_deepl = [tokenizer.tokenize(can) for can in candidates_deepl]

    # sacrebleu = evaluate.load('sacrebleu')
    sacrebleu_google = [sentence_bleu(references, candidate) for candidate in tqdm(candidates_google)]
    sacrebleu_deepl = [sentence_bleu(references, candidate) for candidate in tqdm(candidates_deepl)]

    return get_average(sacrebleu_google) * 100, get_average(sacrebleu_deepl) * 100


if __name__ == '__main__':
    eval_path = '../results/test_tiny_uniform100_inferenced.csv'
    
    # u100_eval = pd.read_csv(eval_path)
    # sentence_num = 200
    # reference = u100_eval['ko'][sentence_num]
    # candidate = u100_eval['deepl_trans'][sentence_num]
    # print(calculate_sentence_bleu(reference, candidate))
    # print(calculate_sentence_rouge(reference, candidate))

    print(calculate_bleu(eval_path))
    print(calculate_sacrebleu(eval_path))
    print(calculate_rouge(eval_path))
    print(calculate_wer(eval_path))
