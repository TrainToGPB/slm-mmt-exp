import os
os.environ['VLLM_USE_MODELSCOPE'] = 'false'
import re
import time
from datetime import datetime

import torch
import streamlit as st
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

LANG_TABLE = {
    'en': 'English',
    'ko': '한국어',
    'ja': '日本語',
    'zh': '中文',
}

@st.cache_resource
def load_model(model_name, vram_limit=0.95, max_length=1024, use_lora=False, **kwargs):
    model = LLM(
        model=model_name,
        tokenizer=model_name,
        gpu_memory_utilization=vram_limit,
        download_dir='/data/.cache/hub/',
        trust_remote_code=True,
        seed=42,
        max_seq_len_to_capture=max_length,
        dtype=torch.bfloat16,
        enable_lora=use_lora,
        max_lora_rank=kwargs.get('max_lora_rank', 16),
    )
    return model

def preprocess_text(input_text, is_structured=False):
    input_text = input_text.strip()

    if '\n' in input_text:
        texts = input_text.split('\n')
        line_breaks = [i for i, text in enumerate(texts) if text.strip() == '']
        texts = [text for text in texts if text.strip() != '']
    else:
        line_breaks = None
        texts = [input_text]

    if is_structured:
        init_symbols = [re.match(r'^[\s\W]+', text).group() if re.match(r'^[\s\W]+', text) else '' for text in texts]
        texts = [text[len(init_simbol):].strip() for text, init_simbol in zip(texts, init_symbols)]
    else:
        init_symbols = None

    return texts, line_breaks, init_symbols

def make_prompt(text, src_lang, tgt_lang):
    instruction = f"Translate from {LANG_TABLE[src_lang]} to {LANG_TABLE[tgt_lang]}."
    src_suffix = f"### {LANG_TABLE[src_lang]}:"
    tgt_suffix = f"### {LANG_TABLE[tgt_lang]}:"
    prompt = f"{instruction}\n{src_suffix} {text}\n{tgt_suffix}"
    return prompt

def postprocess_text(output_texts, line_breaks, init_symbols):
    if init_symbols is not None:
        output_texts = [init_symbol + output_text for init_symbol, output_text in zip(init_symbols, output_texts)]
    if line_breaks is not None:
        for i in line_breaks:
            output_texts.insert(i, '')
    return output_texts

def generate_translation(prompts, model, sampling_params, lora_request=None):
    if lora_request is not None:
        outputs = model.generate(prompts, sampling_params, lora_request=lora_request, use_tqdm=False)
    else:
        outputs = model.generate(prompts, sampling_params, use_tqdm=False)
    return outputs

def translate(input_text, model, sampling_params, lora_request=None, src_lang='ko', tgt_lang='en', is_structured=False):
    input_text = input_text.strip()

    texts_preprocessed, line_breaks, init_symbols = preprocess_text(input_text, is_structured)
    prompts = [make_prompt(text, src_lang, tgt_lang) for text in texts_preprocessed]

    start_time = datetime.now()
    outputs = generate_translation(prompts, model, sampling_params, lora_request)
    end_time = datetime.now()

    outputs_list = [output.outputs[0].text.strip() for output in outputs]
    outputs_list = postprocess_text(outputs_list, line_breaks, init_symbols)

    translation = '\n'.join(outputs_list)
    translation_time = (end_time - start_time).total_seconds() * 1000

    return translation, translation_time

def stream_single(text):
    for paragraph in text.split('  \n'):
        for word in paragraph.split():
            for char in word:
                yield char
                time.sleep(0.01)
            yield ' '
        yield '  \n'

def main():
    st.title("SPARTA-large Translator")

    if 'src_lang' not in st.session_state:
        st.session_state['src_lang'] = 'ko'
    if 'tgt_lang' not in st.session_state:
        st.session_state['tgt_lang'] = 'en'
    if 'adapter' not in st.session_state:
        st.session_state['adapter'] = 'mmt'
    if 'temperature' not in st.session_state:
        st.session_state['temperature'] = 0.0
    if 'top_k' not in st.session_state:
        st.session_state['top_k'] = 40
    if 'top_p' not in st.session_state:
        st.session_state['top_p'] = 0.95
    if 'is_structured' not in st.session_state:
        st.session_state['is_structured'] = False
    if 'stream_text' not in st.session_state:
        st.session_state['stream_text'] = True
    if 'input_text' not in st.session_state:
        st.session_state['input_text'] = ""

    st.sidebar.title("Settings")
    st.sidebar.header("Language")
    st.session_state['src_lang'] = st.sidebar.selectbox("Source Language", options=list(LANG_TABLE.keys()), format_func=lambda x: LANG_TABLE[x], index=1)
    st.session_state['tgt_lang'] = st.sidebar.selectbox("Target Language", options=list(LANG_TABLE.keys()), format_func=lambda x: LANG_TABLE[x], index=0)
    st.sidebar.markdown("---")
    st.sidebar.header("Model Parameters")
    st.session_state['adapter'] = st.sidebar.selectbox("Adapter", ["mmt", "ko-ja", "ko-zh"], placeholder="mmt")
    st.session_state['temperature'] = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0)
    st.session_state['top_k'] = st.sidebar.slider("Top-k", 1, 100, 40)
    st.session_state['top_p'] = st.sidebar.slider("Top-p", 0.1, 1.0, 0.95)
    st.sidebar.markdown("---")
    st.sidebar.header("Others")
    st.session_state['is_structured'] = st.sidebar.checkbox("Structured Text", False)
    st.session_state['stream_text'] = st.sidebar.toggle("Stream Text", True)
    st.sidebar.warning("'Stream Text' option does not support Markdown.")

    model_name = 'beomi/Llama-3-Open-Ko-8B'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    with st.spinner(f"Trying to load translator model..."):
        model = load_model(model_name, 
                           vram_limit=0.95, 
                           max_length=1024, 
                           use_lora=True, 
                           max_lora_rank=64)
    
    sampling_params = SamplingParams(
        temperature=st.session_state['temperature'],
        use_beam_search=False,
        top_k=st.session_state['top_k'],
        top_p=st.session_state['top_p'],
        skip_special_tokens=True,
        stop=tokenizer.eos_token,
        repetition_penalty=1.1,
        max_tokens=1024,
    )
    lora_dict = {
        'mmt': {'name': 'mmt-adapter', 'id': 1, 'path': '/data/sehyeong/nmt/models/mmt_ft/ko-enjazh/ko/'},
        # 'ko-en': {'name': 'koen-adapter', 'id': 2, 'path': '/data/sehyeong/nmt/models/mmt_ft/ko-en/ko/'},
        'ko-ja': {'name': 'koja-adapter', 'id': 3, 'path': '/data/sehyeong/nmt/models/mmt_ft/ko-ja/koja-halfja/'},
        'ko-zh': {'name': 'kozh-adapter', 'id': 4, 'path': '/data/sehyeong/nmt/models/mmt_ft/ko-zh/ko/'},
    }

    st.header("Translate Text")
    st.session_state['input_text'] = st.text_area("", "", height=400, placeholder="Type anything you want to translate...")
    
    if st.button("Translate"):
        if st.session_state['input_text'].strip() == "":
            st.warning("Please enter some text to translate.")
        else:
            with st.spinner("Translating..."):
                lora_request = LoRARequest(lora_dict[st.session_state['adapter']]['name'], lora_dict[st.session_state['adapter']]['id'], lora_dict[st.session_state['adapter']]['path'])
                try:
                    translation, translation_time = translate(st.session_state['input_text'], 
                                                              model, 
                                                              sampling_params, 
                                                              lora_request, 
                                                              src_lang=st.session_state['src_lang'], 
                                                              tgt_lang=st.session_state['tgt_lang'], 
                                                              is_structured=st.session_state['is_structured'])
                except Exception as e:
                    st.error(f"An error occurred during translation: {e}")
                    return
                
                translation_for_print = '  \n'.join(translation.split('\n'))

                st.success(f"Translation completed in {translation_time:.2f} ms")
                if st.session_state['stream_text']:
                    st.write_stream(stream_single(text=translation_for_print))
                else:
                    st.write(translation_for_print)

            print(st.session_state['input_text'])
            print(translation_for_print)

if __name__ == "__main__":
    main()
