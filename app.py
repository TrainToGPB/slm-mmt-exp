import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import re
import time
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import streamlit as st
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig


DEVICE = torch.cuda.current_device()


@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer():
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

    tokenizer = AutoTokenizer.from_pretrained(plm_name, trust_remote_code=True)
    tokenizer.pad_token = "</s>"
    tokenizer.pad_token_id = 2
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.eos_token_id = 46332
    tokenizer.add_eos_token = True
    tokenizer.padding_side = 'right'
    tokenizer.model_max_length = 768

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

    outputs = model.generate(**inputs, max_length=768, eos_token_id=46332)

    translation = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    translation = re.sub(r'\s+', ' ', translation)
    translation = translation.strip()

    return translation


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

    st.sidebar.title("기능 선택")
    selected_option = st.sidebar.radio("__원하는 작업을 선택하세요__", options=["단일 문장 번역", "CSV 파일 번역"])

    if selected_option == "단일 문장 번역":
        st.write_stream(stream_single)

        text = st.text_area("영어 문장 입력")

        if st.button("번역"):
            if text.strip() != "":
                model, tokenizer = load_model_and_tokenizer()
                
                start_time = datetime.now()
                translation = translate(text, model, tokenizer)
                end_time = datetime.now()
                translation_time = (end_time - start_time).total_seconds()

                st.write("__번역 결과:__")
                st.write(translation)
                st.write("__번역 소요 시간:__")
                st.write(f"{translation_time:.4f} 초")
            else:
                st.warning("문장을 입력해주세요.")

    elif selected_option == "CSV 파일 번역":
        st.write_stream(stream_csv)

        file = st.file_uploader("CSV 파일 업로드", type=['csv'])

        if file is not None:
            df = pd.read_csv(file)

            st.write("업로드한 파일 미리보기:")
            st.write(df)

            columns = df.columns.tolist()
            selected_column = st.selectbox("번역할 열을 선택하세요:", columns)

            if st.button("번역"):
                model, tokenizer = load_model_and_tokenizer()

                translations = []
                start_time = datetime.now()
                for text in df[selected_column]:
                    translation = translate(text, model, tokenizer)
                    translations.append(translation)
                end_time = datetime.now()
                translation_time = (end_time - start_time).total_seconds()
                mean_translation_time = translation_time / len(translations)

                translation_column = 'translated_text'
                col_num = 1
                while translation_column in columns:
                    translation_column = f'translated_text_{col_num}'
                    col_num += 1

                df[translation_column] = translations

                st.wrtie("__번역된 파일 미리보기:__")
                st.write(df)
                st.wrtie("__총 번역 소요 시간:__")
                st.write(f"{translation_time:.4f} 초")
                st.wrtie("__문장 당 평균 번역 소요 시간:__")
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

