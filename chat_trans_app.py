import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from collections import defaultdict

import torch
import streamlit as st
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer


CHAT_DEVICE = torch.device('cuda:0')
TRANS_DEVICE = torch.device('cuda:0')
LANG_TABLE = {
    'en': 'English',
    'ko': '한국어',
    'ja': '日本語',
    'zh': '中文',
}


@st.cache_resource
def load_model(model_name, vram_limit=0.475, max_length=4000, use_lora=False, **kwargs):
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


@st.cache_resource
def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def load_sampling_params(temperature=0.0,
                         top_k=40,
                         top_p=0.95,
                         stop='<|end_of_text|>',
                         repetition_penalty=1.0,
                         max_tokens=4000):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        skip_special_tokens=True,
        stop=stop,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
    )
    return sampling_params


def get_model_response(prompt, model, sampling_params, lora_path=None, lora_name=None):
    if lora_path is not None:
        if lora_name is None:
            lora_name = 'adapter1'
        lora_request = LoRARequest(lora_name, 1, lora_path)
    else:
        lora_request = None

    outputs = model.generate(prompt, sampling_params, lora_request=lora_request, use_tqdm=False)
    response = outputs[0].outputs[0].text

    return response


def postprocess_chat_response(response):
    if response.split('\n')[0] == '<|start_header_id|>assistant<|end_header_id|>':
        response = response.split('\n', 1)[1].strip()
    return response



def init_session_state():
    defaults = {
        'chat_model_name': None,
        'chat_lora_name': None,
        'chat_lora_nickname': None,
        'chat_model': None,
        'chat_tokenizer': None,
        'chat_sampling_params': None,
        'trans_model_name': None,
        'trans_lora_name': None,
        'trans_lora_nickname': None,
        'trans_model': None,
        'trans_sampling_params': None,
        'messages': defaultdict(list),
        'display_lang': 'en',
        'adapter': 'None',
        'temperature': 0.7,
        'top_k': 40,
        'top_p': 0.95,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    return st.session_state


def setup_sidebar(session_state):
    st.sidebar.title("Settings")

    if st.sidebar.button("Clear Chat"):
        session_state.messages = defaultdict(list)
    st.sidebar.markdown("---")
    
    st.sidebar.header("Language")
    session_state['display_lang'] = st.sidebar.selectbox("Display Language", 
                                                         options=list(LANG_TABLE.keys()), 
                                                         format_func=lambda x: LANG_TABLE[x], 
                                                         index=0)
    st.sidebar.markdown("---")

    st.sidebar.header("Chat Model Parameters")
    session_state['temperature'] = st.sidebar.slider("Temperature", 0.7, 1.0, 0.0)
    session_state['top_k'] = st.sidebar.slider("Top-k", 1, 100, 40)
    session_state['top_p'] = st.sidebar.slider("Top-p", 0.1, 1.0, 0.95)
    st.sidebar.markdown("---")

    return session_state


def setup_chat_model(session_state):
    session_state.chat_model = load_model(session_state.chat_model_name, use_lora=True)
    session_state.chat_tokenizer = load_tokenizer(session_state.chat_model_name)
    session_state.chat_sampling_params = load_sampling_params(
        temperature=session_state.temperature,
        top_k=session_state.top_k,
        top_p=session_state.top_p,
        stop=['<|eot_id|>', '<|end_of_text|>']
    )
    return session_state


def setup_trans_model(session_state):
    session_state.trans_model = load_model(session_state.trans_model_name, use_lora=True)
    session_state.trans_tokenizer = load_tokenizer(session_state.trans_model_name)
    session_state.trans_sampling_params = load_sampling_params(
        temperature=0.0,
        top_k=40,
        top_p=0.95,
        repetition_penalty=1.1,
        stop=['<|eot_id|>', '<|end_of_text|>']
    )
    return session_state


def display_chat(session_state):
    for message in session_state.messages[session_state.display_lang]:
        with st.chat_message(message['role']):
            st.markdown(message['content'])


def chat_in_english(session_state):
    chat_prompt = session_state.chat_tokenizer.apply_chat_template(session_state.messages['en'], tokenize=False)
    response = get_model_response(chat_prompt, 
                                    session_state.chat_model, 
                                    session_state.chat_sampling_params)
    response = postprocess_chat_response(response)
    return response, session_state


def translate(prompt, src_lang, tgt_lang, session_state):
    instruction = f"Translate this from {src_lang} to {tgt_lang}."
    src_suffix = f"### {src_lang}:"
    tgt_suffix = f"### {tgt_lang}:"
    trans_prompt = f"{instruction}\n{src_suffix} {prompt}\n{tgt_suffix}"
    translation = get_model_response(trans_prompt, 
                                     session_state.trans_model,
                                     session_state.trans_sampling_params,
                                     lora_path=session_state.trans_lora_name,
                                     lora_name=session_state.trans_lora_nickname)
    return translation


def setup_trans_chat(session_state):
    chat_prompt_dict = {'role': 'user', 'content': prompt} # ???
    session_state.messages[session_state.display_lang].append(chat_prompt_dict)
    
    if prompt := st.chat_input("Ask anything..."):
        with st.chat_message('user'):
            st.markdown(prompt)
        if session_state.display_lang == 'en':
            prompt_to_input = prompt
        else:
            prompt_to_input = translate(prompt, session_state)
        response, session_state = chat_in_english(prompt_to_input, session_state)
        with st.chat_message('assistant'):
            st.markdown(response)

        session_state.messages[session_state.display_lang].append({'role': 'assistant', 'content': response})

    return session_state


def main():
    st.title("Chat with Translation")
    session_state = init_session_state()
    session_state = setup_sidebar(session_state)

    if session_state.chat_model_name is None:
        session_state.chat_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    session_state.chat_lora_name = None
    session_state.chat_lora_nickname = None
    session_state = setup_chat_model(session_state)

    if session_state.trans_model_name is None:
        session_state.trans_model_name = 'beomi/Llama-3-Open-Ko-8B'
    session_state.trans_lora_name = '/data/sehyeong/nmt/models/mmt_ft/ko-enjazh/ko'
    session_state.trans_lora_nickname = 'mmt-ko'
    session_state = setup_trans_model(session_state)

    display_chat(session_state)

    session_state = setup_trans_chat(session_state)


if __name__ == '__main__':
    main()
