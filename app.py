import os
os.environ['VLLM_USE_MODELSCOPE'] = 'false'
import re
import time
from datetime import datetime

import torch
import pandas as pd
import streamlit as st
from vllm import LLM, SamplingParams
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM
from transformers import BitsAndBytesConfig


FULL_LANG = {
    'en': 'English',
    'ko': '한국어',
}


@st.cache_resource
def load_qlora_model(gpu_id=0):
    DEVICE = torch.device(f'cuda:{gpu_id}')

    plm_name = 'beomi/open-llama-2-ko-7b'
    adapter_path = 'traintogpb/llama-2-en2ko-translator-7b-qlora-adapter'

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
        torch_dtype=torch_dtype,
        device_map={"": DEVICE}
    )
    model = PeftModel.from_pretrained(
        model, 
        adapter_path, 
        torch_dtype=torch_dtype
    )

    return model, adapter_path, gpu_id


@st.cache_resource
def load_bf16_model(gpu_id=1):
    DEVICE = torch.device(f'cuda:{gpu_id}')

    model_path = 'traintogpb/llama-2-en2ko-translator-7b-qlora-bf16-upscaled'
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map={"": DEVICE})
    model.to(DEVICE)
    return model, model_path, gpu_id


@st.cache_resource
def load_bf16_vllm_model(gpu_id=2):
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu_id}'
    model_path = 'traintogpb/llama-2-en2ko-translator-7b-qlora-bf16-upscaled'
    model = LLM(model_path, seed=42)
    return model, model_path, gpu_id


@st.cache_resource
def load_qlora_sparta_model(gpu_id=0):
    DEVICE = torch.device(gpu_id)

    plm_name = 'beomi/open-llama-2-ko-7b'
    adapter_path = 'traintogpb/llama-2-enko-translator-7b-qlora-adapter'

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
        torch_dtype=torch_dtype,
        device_map={"": DEVICE}
    )
    model = PeftModel.from_pretrained(
        model, 
        adapter_path, 
        torch_dtype=torch_dtype
    )

    return model, adapter_path, gpu_id


@st.cache_resource
def load_bf16_sparta_model(gpu_id=1):
    DEVICE = torch.device(f'cuda:{gpu_id}')

    model_path = 'traintogpb/llama-2-enko-translator-7b-qlora-bf16-upscaled'
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map={"": DEVICE})
    model.to(DEVICE)
    return model, model_path, gpu_id


@st.cache_resource
def load_bf16_vllm_sparta_model(gpu_id=2):
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu_id}'
    model_path = 'traintogpb/llama-2-enko-translator-7b-qlora-bf16-upscaled'
    model = LLM(model_path, seed=42)
    return model, model_path, gpu_id


def load_model(model_type, gpu_id):
    model_dict = {
        'qlora': load_qlora_model,
        'bf16-upscaled': load_bf16_model,
        'bf16-upscaled-vllm': load_bf16_vllm_model,
        'qlora-sparta': load_qlora_sparta_model,
        'bf16-upscaled-sparta': load_bf16_sparta_model,
        'bf16-upscaled-vllm-sparta': load_bf16_vllm_sparta_model,
        'llama-sparta': load_bf16_vllm_sparta_model,
    }
    load_model_func = model_dict[model_type.lower()]
    return load_model_func(gpu_id)


def load_model_and_tokenizer(_model_type):
    gpu_id = 0
    if 'bf16' in _model_type.lower():
        if 'vllm' not in _model_type.lower():
            gpu_id = 1
        else:
            gpu_id = 2
            
    with st.spinner(f"{_model_type} 모델 불러오는 중..."):
        model, model_path, gpu_id = load_model(_model_type, gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = "</s>"
    tokenizer.pad_token_id = 2
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.eos_token_id = 46332
    tokenizer.add_eos_token = True
    tokenizer.padding_side = 'right'
    tokenizer.model_max_length = 768

    return model, tokenizer, gpu_id


def translate(text, selected_model, model, tokenizer, gpu_id, src_lang='en', tgt_lang='ko'):
    if text is None or text == "" or pd.isna(text):
        text = " "
    if selected_model.lower().endswith('sparta'):
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        text_formatted = f"Translate this from {FULL_LANG[src_lang]} to {FULL_LANG[tgt_lang]}.\n### {FULL_LANG[src_lang]} {text}\n### {FULL_LANG[tgt_lang]}:"
    else:
        text_formatted = f"### {src_lang}: {text}\n### {tgt_lang}: "
    
    if isinstance(model, LlamaForCausalLM) or isinstance(model, PeftModel):
        inputs = tokenizer(
            text_formatted,
            return_tensors='pt',
            max_length=768,
            padding=True,
            truncation=True
        )
        device = torch.device(f'cuda:{gpu_id}')
        inputs = {key: value.to(device) for key, value in inputs.items()}
        inputs['input_ids'] = inputs['input_ids'][0][:-1].unsqueeze(dim=0)
        inputs['attention_mask'] = inputs['attention_mask'][0][:-1].unsqueeze(dim=0)
        input_len = len(inputs['input_ids'].squeeze())

        start_time = datetime.now()
        outputs = model.generate(**inputs, max_length=768, eos_token_id=46332)
        end_time = datetime.now()

        translation = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    elif isinstance(model, LLM):
        sampling_params = SamplingParams(
            best_of=1,
            temperature=0, 
            top_p=0.95,
            top_k=40,
            skip_special_tokens=True,
            stop='<|endoftext|>',
            frequency_penalty=0.0,
            repetition_penalty=1.1,
            max_tokens=350
        )
        start_time = datetime.now()
        outputs = model.generate([text_formatted], sampling_params, use_tqdm=False)
        end_time = datetime.now()
        
        translation = outputs[0].outputs[0].text

    else:
        raise ValueError("Invalid model type")

    translation_time = (end_time - start_time).total_seconds()

    translation = re.sub(r'\s+', ' ', translation)
    translation = translation.strip()

    return translation, translation_time


def translate_csv(df, selected_column, selected_model, model, tokenizer, gpu_id, direction=None):
    translations = []
    translation_times = []

    progress_text = "번역 중..."
    percent_complete = 0
    progress_bar = st.progress(percent_complete, text=progress_text)
    total_rows = len(df)
    for _, row in df.iterrows():
        text = row[selected_column]
        if direction is not None:
            src_lang, tgt_lang = direction.split('-')
        else:
            src_lang, tgt_lang = 'en', 'ko'
        translation, translation_time = translate(text, selected_model, model, tokenizer, gpu_id, src_lang, tgt_lang)

        translations.append(translation)
        translation_times.append(translation_time)

        percent_complete += 1 / total_rows
        progress_bar.progress(percent_complete, text=progress_text)

    progress_bar.empty()

    return translations, translation_times


def stream_single():
    guide_text = "입력된 문장을 번역합니다." 
    for word in guide_text.split():
        for char in word:
            yield char
            time.sleep(0.05)
        yield ' '


def stream_csv():
    guide_text = "CSV 파일을 업로드하여 파일 내의 문장을 번역합니다." 
    for word in guide_text.split():
        for char in word:
            yield char
            time.sleep(0.05)
        yield ' '


def main():
    st.title("EN-KO Translator")
    st.write("__문장을 입력하시면 번역해 드립니다!__")

    # selected_model = st.radio("번역할 모델을 선택하세요:", ["QLoRA", 
    #                                                       "BF16-Upscaled", 
    #                                                       "BF16-Upscaled-vLLM", 
    #                                                       "QLoRA-Sparta",
    #                                                       "BF16-Upscaled-Sparta",
    #                                                       "BF16-Upscaled-vLLM-Sparta"])
    selected_model = st.radio("번역할 모델을 선택하세요:", ["LLaMA-Sparta"])

    st.sidebar.title("기능 선택")
    selected_option = st.sidebar.radio("__원하는 작업을 선택하세요__", options=["단일 문장 번역", "CSV 파일 번역"])

    if selected_option == "단일 문장 번역":
        st.write_stream(stream_single) # 입력된 문장을 번역합니다.

        if 'sparta' in selected_model.lower():
            direction = st.radio("번역 방향을 선택하세요:", ["한국어 → 영어", "영어 → 한국어"])
            if direction == "한국어 → 영어":
                src_lang, tgt_lang = 'ko', 'en'
            else:
                src_lang, tgt_lang = 'en', 'ko'

            if src_lang == 'en':
                text = st.text_area("영어 문장 입력")
            elif src_lang == 'ko':
                text = st.text_area("한국어 문장 입력")
        
        else:
            src_lang, tgt_lang = 'en', 'ko'
            text = st.text_area("영어 문장 입력")

        if st.button("번역"):
            if text.strip() != "":
                print(selected_model.upper())
                model, tokenizer, gpu_id = load_model_and_tokenizer(selected_model)
                translation, translation_time = translate(text, selected_model, model, tokenizer, gpu_id, src_lang, tgt_lang)

                st.write("__번역 결과:__")
                st.write(translation)
                st.write("__번역 소요 시간:__")
                st.write(f"{translation_time:.4f} 초")
            else:
                st.warning("문장을 입력해주세요.")

    elif selected_option == "CSV 파일 번역":
        st.write_stream(stream_csv) # CSV 파일을 업로드하여 파일 내의 문장을 한글로 번역합니다.

        file = st.file_uploader("CSV 파일 업로드", type=['csv'])

        if file is not None:
            df = pd.read_csv(file)
            
            st.write("업로드한 파일 미리보기:")
            st.write(df)

            if 'sparta' in selected_model.lower():
                direction = st.radio("번역 방향을 선택하세요:", ["한국어 → 영어", "영어 → 한국어", "파일 내 'direction' 칼럼 있음"])
                if direction == "한국어 → 영어":
                    direction = 'ko-en'
                elif direction == "영어 → 한국어":
                    direction = 'en-ko'
                else:
                    direction = None

            columns = df.columns.tolist()
            selected_column = st.selectbox("번역할 열을 선택하세요:", columns)

            if st.button("번역"):
                print(selected_model.upper())
                model, tokenizer, gpu_id = load_model_and_tokenizer(selected_model)
                translations, translation_times = translate_csv(df, selected_column, selected_model, model, tokenizer, gpu_id, direction)
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
