import os
import re
import multiprocessing
from functools import partial
from datetime import datetime

from kss import split_sentences
import torch
torch.multiprocessing.set_start_method('spawn')

from model_inferences import load_model_and_tokenizer, inference_single


def preprocess_text(text):
    text = re.sub(r'"', '\"', text)
    text = re.sub(r"'", "\'", text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def split_text(text):
    len_text = len(text)
    if len_text > 1500:
        raise ValueError("Text length should be less than 1500 characters.")
    len_threshold = len_text // 4
    
    splits = split_sentences(text, backend='Mecab')
    chunks = []
    chunk_tmp = []
    chunk_len = 0
    for idx, split in enumerate(splits):
        chunk_len += len(split)
        chunk_tmp.append(split)
        if chunk_len > len_threshold or idx == len(splits) - 1:
            chunks.append(' '.join(chunk_tmp))
            chunk_tmp = []
            chunk_len = 0

    return chunks


def translate_partial(model_type, model, tokenizer, partial_texts, src_lang, tgt_lang, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    translations = []
    for part in partial_texts:
        translation = inference_single(model_type, model, tokenizer, part, src_lang, tgt_lang)
        translations.append(translation)
    return translations


def translate_parallel(model_type, text, src_lang, tgt_lang, num_processes=4):
    partial_texts = split_text(text)
    model, tokenizer = load_model_and_tokenizer(model_type)

    pool = multiprocessing.Pool(processes=num_processes)
    gpu_ids = list(range(num_processes))
    func = partial(translate_partial, model_type=model_type, model=model, tokenzier=tokenizer, src_lang=src_lang, tgt_lang=tgt_lang)
    results = pool.starmap(func, zip(partial_texts, gpu_ids))
    translations = [translation for result in results for translation in result]
    pool.close()
    pool.join()

    translated_text = ' '.join(translations)
    
    return translated_text


if __name__ == '__main__':
    text = """
    티맥스그룹은 인공지능(AI) 플랫폼·서비스 전문 계열사 티맥스AI가 육군 규정 교육 챗봇 시스템 '하이퍼 챗봇' 최종 버전을 공개했다고 15일 밝혔다.

    티맥스AI는 지난 12일부터 이틀간 대전컨벤션센터(DCC)에서 진행된 '2023년 민·군 기술협력 사업 성과발표회'에 참석해 성과물 전시장에서 과제 결과물인 하이퍼 챗봇의 최종 버전을 선보였다. 티맥스AI는 2022년 6월부터 민·군 기술협력 사업으로 육군 규정 교육 챗봇 시스템 구현 과제를 수행 중이며 지난해 10월 한국전자전(KES 2023)에서 데모 버전을 선보였다.

    하이퍼 챗봇은 복잡한 육군 규정을 효율적으로 학습하고 검색할 수 있도록 개발한 대화형 AI 챗봇 시스템이다. 육군 규정집을 자동 파싱(Parsing·구문 분석)해데이터 베이스에 적재하는 방식으로 규정집 전체를 정확하게 검색하도록 구현했다.

    규정 챗봇용 대화 말뭉치를 확보해 질의응답 기능도 갖췄다. 버튼형 질의응답으로 군 복무자 누구나 육군 규정 정보를 쉽고 빠르게 검색할 수 있다.

    전시장에는 참석자들이 기술을 시연해 볼 수 있는 부스가 설치됐다.

    티맥스AI는 민·군 기술협력 사업 수행 내역을 바탕으로, 각 분야에 적용할 수 있는 대화형 AI 챗봇시스템 개발도 마쳤다. 'A-Talk'(에이-톡)이라는 이름으로 육군 이외 금융·법률·공공기관 등 다양한 분야에서 활용될 예정이다.
    """
    model_type = 'llama-sparta-qlora-bf16-vllm'
    src_lang, tgt_lang = 'ko', 'en'

    start_time = datetime.now()
    translated_text = translate_parallel(model_type, text, src_lang, tgt_lang)
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    print(f"Translated text: {translated_text}")
    print(f"Elapsed time: {elapsed_time} seconds")
