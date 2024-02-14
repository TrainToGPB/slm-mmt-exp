import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import re
import time
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import streamlit as st
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig


DEVICE = torch.cuda.current_device()


@st.cache_resource
def load_model_and_tokenizer(_model_type):
    with st.spinner("모델 불러오는 중..."):
        if _model_type == "QLoRA":
            plm_name = 'beomi/open-llama-2-ko-7b'
            lora_path = './training/llama_qlora/models/baseline'
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
            torch_dtype = torch.bfloat16
            model = AutoModelForCausalLM.from_pretrained(
                plm_name, 
                quantization_config=bnb_config, 
                torch_dtype=torch_dtype
            )
            model = PeftModel.from_pretrained(
                model, 
                lora_path, 
                torch_dtype=torch_dtype
            )
            model.to(torch.cuda.current_device())

        elif _model_type == "BF16-Upscaled":
            plm_name = './training/llama_qlora/models/baseline-merged-bf16'
            model = AutoModelForCausalLM.from_pretrained(plm_name, torch_dtype=torch.float16)

        model.to(torch.cuda.current_device())

        tokenizer = AutoTokenizer.from_pretrained(plm_name)
        tokenizer.pad_token = "</s>"
        tokenizer.pad_token_id = 2
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.eos_token_id = 46332
        tokenizer.add_eos_token = True
        tokenizer.padding_side = 'right'
        tokenizer.model_max_length = 768

    # st.success("모델 불러오기 완료!")

    return model, tokenizer


def translate(text, model, tokenizer):
    if text is None or text == "" or pd.isna(text):
        text = " "
    text_formatted = f"### English: {text}\n### 한국어: "

    inputs = tokenizer(
        text_formatted,
        return_tensors='pt',
        max_length=768,
        padding=True,
        truncation=True
    )
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    inputs['input_ids'] = inputs['input_ids'][0][:-1].unsqueeze(dim=0)
    inputs['attention_mask'] = inputs['attention_mask'][0][:-1].unsqueeze(dim=0)
    input_len = len(inputs['input_ids'].squeeze())

    start_time = datetime.now()
    outputs = model.generate(**inputs, max_length=768, eos_token_id=46332)
    end_time = datetime.now()
    translation_time = (end_time - start_time).total_seconds()

    translation = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    translation = re.sub(r'\s+', ' ', translation)
    translation = translation.strip()

    return translation, translation_time


def translate_csv(file, selected_model):
    model, tokenizer = load_model_and_tokenizer(selected_model)

    translations = []
    translation_times = []

    progress_text = "번역 중..."
    percent_complete = 0
    progress_bar = st.progress(percent_complete, text=progress_text)
    total_rows = len(file)
    for text in file:
        translation, translation_time = translate(text, model, tokenizer)

        translations.append(translation)
        translation_times.append(translation_time)

        percent_complete += 1 / total_rows
        progress_bar.progress(percent_complete, text=progress_text)

    progress_bar.empty()

    return translations, translation_times


def stream_single():
    guide_text = "입력된 한글 문장을 영어로 번역하는 시스템입니다." 
    for word in guide_text.split():
        for char in word:
            yield char
            time.sleep(0.05)
        yield ' '


def stream_csv():
    guide_text = "CSV 파일을 업로드하여 파일 내의 영어 문장을 한글로 번역하는 시스템입니다." 
    for word in guide_text.split():
        for char in word:
            yield char
            time.sleep(0.05)
        yield ' '


def main():
    st.title("EN-KO Translator")
    st.write("__영어 문장을 입력하시면, 한글로 번역해 드립니다!__")

    st.write("번역할 모델을 선택하세요:")
    selected_model = st.radio("", ["QLoRA", "BF16-Upscaled"])

    st.sidebar.title("기능 선택")
    selected_option = st.sidebar.radio("__원하는 작업을 선택하세요__", options=["단일 문장 번역", "CSV 파일 번역"])

    if selected_option == "단일 문장 번역":
        st.write_stream(stream_single) # 입력된 한글 문장을 영어로 번역하는 시스템입니다.

        text = st.text_area("영어 문장 입력")

        if st.button("번역"):
            if text.strip() != "":
                model, tokenizer = load_model_and_tokenizer(selected_model)

                translation, translation_time = translate(text, model, tokenizer)

                st.write("__번역 결과:__")
                st.write(translation)
                st.write("__번역 소요 시간:__")
                st.write(f"{translation_time:.4f} 초")
            else:
                st.warning("문장을 입력해주세요.")

    elif selected_option == "CSV 파일 번역":
        st.write_stream(stream_csv) # CSV 파일을 업로드하여 파일 내의 영어 문장을 한글로 번역하는 시스템입니다.

        file = st.file_uploader("CSV 파일 업로드", type=['csv'])

        if file is not None:
            df = pd.read_csv(file)
            
            st.write("업로드한 파일 미리보기:")
            st.write(df)

            columns = df.columns.tolist()
            selected_column = st.selectbox("번역할 열을 선택하세요:", columns)

            if st.button("번역"):
                model, tokenizer = load_model_and_tokenizer(selected_model)

                translations, translation_times = translate_csv(df[selected_column], model)
                mean_translation_time = sum(translation_times) / len(translation_times)

                translation_column = 'translated_text'
                col_num = 1
                while translation_column in columns:
                    translation_column = f'translated_text_{col_num}'
                    col_num += 1
                df[translation_column] = translations

                translation_time_column = 'translation_time'
                df[translation_time_column] = translation_times

                st.write("__번역된 파일 미리보기:__")
                st.write(df)
                st.write("__총 번역 소요 시간:__")
                st.write(f"{sum(translation_times):.4f} 초")
                st.write("__문장 당 평균 번역 소요 시간:__")
                st.write(f"{mean_translation_time:.4f} 초")

                translated_file = df.to_csv(index=False, encoding="utf-8-sig")
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_file_name = f"translated_file_{current_time}.csv"
                st.download_button(
                    label="번역된 파일 다운로드",
                    data=translated_file,
                    file_name=save_file_name,
                    mime='text/csv'
                )


if __name__ == "__main__":
    main()

