import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import re

import torch
import streamlit as st
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig


DEVICE = torch.cuda.current_device()


@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer():
    plm_name = 'beomi/open-llama-2-ko-7b'
    lora_path = '../../training/llama_qlora/models/baseline'
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


def main():
    st.title("EN-KO Translator")
    st.write("영어 문장을 입력하시면, 한글로 번역합니다.")

    text = st.text_area("번역할 영어 문장을 입력하세요.")

    if st.button("번역"):
        if text.strip() != "":
            model, tokenizer = load_model_and_tokenizer()
            translation = translate(text, model, tokenizer)
            st.write("번역 결과:")
            st.write(translation)
        else:
            st.warning("문장을 입력해주세요.")


if __name__ == "__main__":
    main()

