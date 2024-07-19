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


MODEL_NAME = st.secrets["model"]["name"]
VRAM_LIMIT = st.secrets["model"]["vramLimit"]
MAX_LENGTH = st.secrets["model"]["maxLength"]
USE_LORA = st.secrets["model"]["useLora"]
MAX_LORA_RANK = st.secrets["model"]["maxLoraRank"]
LANG_TABLE = {
    'en': 'English',
    'ko': '한국어',
    'ja': '日本語',
    'zh': '中文',
}
LORA_DICT = {
    'mmt': {
        'name': 'mmt-adapter', 
        'id': 1, 'path': 
        '/data/sehyeong/nmt/models/mmt_ft/ko-enjazh/ko/'
    },
    'ko-ja': {
        'name': 'koja-adapter', 
        'id': 3, 
        'path': '/data/sehyeong/nmt/models/mmt_ft/ko-ja/koja-halfja/'
    },
    'ko-zh': {
        'name': 'kozh-adapter', 
        'id': 4, 
        'path': '/data/sehyeong/nmt/models/mmt_ft/ko-zh/ko/'
    },
}


def initialize_session_state():
    defaults = {
        'src_lang': 'ko',
        'tgt_lang': 'en',
        'adapter': 'mmt',
        'temperature': 0.0,
        'top_k': 40,
        'top_p': 0.95,
        'is_structured': False,
        'stream_text': True,
        'input_text': ""
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    return st.session_state

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

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = load_model(model_name, vram_limit=VRAM_LIMIT, max_length=MAX_LENGTH, use_lora=USE_LORA, max_lora_rank=MAX_LORA_RANK)
    return model, tokenizer

def load_sampling_params(tokenizer):
    sampling_params = SamplingParams(
        temperature=st.session_state['temperature'],
        use_beam_search=False,
        top_k=st.session_state['top_k'],
        top_p=st.session_state['top_p'],
        skip_special_tokens=True,
        stop=tokenizer.eos_token,
        repetition_penalty=st.secrets["model"]["repetitionPenalty"],
        max_tokens=st.secrets["model"]["maxTokensToGenerate"],
    )
    return sampling_params

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

def setup_translate(model, lora_dict, sampling_params, session_state):
    if session_state['input_text'].strip() == "":
        st.warning("Please enter some text to translate.")
    else:
        with st.spinner("Translating..."):
            lora_request = LoRARequest(lora_dict[session_state['adapter']]['name'], 
                                       lora_dict[session_state['adapter']]['id'], 
                                       lora_dict[session_state['adapter']]['path'])
            try:
                translation, translation_time = translate(session_state['input_text'], 
                                                          model, 
                                                          sampling_params, 
                                                          lora_request, 
                                                          src_lang=session_state['src_lang'], 
                                                          tgt_lang=session_state['tgt_lang'], 
                                                          is_structured=session_state['is_structured'])
                session_state['translation_for_print'] = '  \n'.join(translation.split('\n'))
                if session_state['stream_text']:
                    st.write_stream(stream_single(text=session_state['translation_for_print']))
                else:
                    st.text_area(label=f"Translated {LANG_TABLE[session_state['tgt_lang']]} text", 
                                 value=session_state['translation_for_print'],
                                 height=400)
                st.success(f"Translation completed in {translation_time:.2f} ms")
            except Exception as e:
                st.error(f"An error occurred during translation: {e}")
                
    return session_state

def stream_single(text):
    for paragraph in text.split('  \n'):
        for word in paragraph.split():
            for char in word:
                yield char
                time.sleep(0.01)
            yield ' '
        yield '  \n'

def setup_sidebar(session_state):
    st.sidebar.title("Settings")

    st.sidebar.header("Language")
    session_state['src_lang'] = st.sidebar.selectbox("Source Language", options=list(LANG_TABLE.keys()), format_func=lambda x: LANG_TABLE[x], index=1)
    session_state['tgt_lang'] = st.sidebar.selectbox("Target Language", options=list(LANG_TABLE.keys()), format_func=lambda x: LANG_TABLE[x], index=0)
    st.sidebar.markdown("---")

    st.sidebar.header("Model Parameters")
    session_state['adapter'] = st.sidebar.selectbox("Adapter", ["mmt", "ko-ja", "ko-zh"], placeholder="mmt")
    session_state['temperature'] = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0)
    session_state['top_k'] = st.sidebar.slider("Top-k", 1, 100, 40)
    session_state['top_p'] = st.sidebar.slider("Top-p", 0.1, 1.0, 0.95)
    st.sidebar.markdown("---")

    st.sidebar.header("Others")
    session_state['is_structured'] = st.sidebar.checkbox("Structured Text", True)
    session_state['stream_text'] = st.sidebar.toggle("Stream Text", False)
    st.sidebar.warning("'Stream Text' option does not support Markdown.")

    return session_state

def input_area(session_state):
    session_state['input_text'] = st.text_area("", "", height=400, placeholder="Type anything you want to translate...")
    return session_state

def setup_ui(session_state):
    session_state = setup_sidebar(session_state)
    st.header("Translate Text")
    session_state = input_area(session_state)
    return session_state

def main():
    st.title("SPARTA-large Translator")
    st.session_state = initialize_session_state()
    st.session_state = setup_ui(st.session_state)

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    sampling_params = load_sampling_params(tokenizer)
    lora_dict = {
        'mmt': {'name': 'mmt-adapter', 'id': 1, 'path': '/data/sehyeong/nmt/models/mmt_ft/ko-enjazh/ko/'},
        'ko-ja': {'name': 'koja-adapter', 'id': 3, 'path': '/data/sehyeong/nmt/models/mmt_ft/ko-ja/koja-halfja/'},
        'ko-zh': {'name': 'kozh-adapter', 'id': 4, 'path': '/data/sehyeong/nmt/models/mmt_ft/ko-zh/ko/'},
    }

    if st.button("Translate"):
        st.session_state = setup_translate(model, lora_dict, sampling_params, st.session_state)

if __name__ == "__main__":
    main()
