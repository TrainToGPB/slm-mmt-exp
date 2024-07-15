import gc
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys

import torch
import streamlit as st
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.distributed.parallel_state import destroy_model_parallel
from transformers import AutoConfig


CHAT_DEVICE = torch.device('cuda:0')
TRANS_DEVICE = torch.device('cuda:0')


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


def clear_chat_model(chat_model, session_state):
    destroy_model_parallel()
    print("############ DESTROY MODEL PARALLEL ############")
    if 'chat_model' in session_state:
        session_state.chat_model = ""

    try:
        del chat_model.llm_engine.model_executor.driver_worker
        print("############ DELETE CHAT MODEL ############")
    except:
        print("############ NO CHAT MODEL TO BE DELETED ############")
    try:
        del chat_model
    except:
        pass

    gc.collect()
    torch.cuda.empty_cache()
    print("############ EMPTY VRAM CACHE ############")


def main():
    st.title("Chat AI Test for Tmax CA2 1-3")

    with st.sidebar:
        clear_chat = st.button('Clear Chat')
        trans_to_english = st.toggle('Non-English', False)

    chat_model = None
    if 'current_model_name' not in st.session_state:
        st.session_state.current_model_name = ""
    if 'chat_model' not in st.session_state:
        st.session_state.chat_model = chat_model
    with st.spinner(f"Trying to load translator model..."):
        trans_model = load_model('beomi/Llama-3-Open-Ko-8B',
                                 vram_limit=0.40,
                                 max_length=8192,
                                 use_lora=True,
                                 max_lora_rank=64)
        st.session_state.trans_model = trans_model

    chat_model_name = st.text_input("Input the HuggingFace chat model name", placeholder="meta-llama/Meta-Llama-3-8B-Instruct")
    if not chat_model_name:
        chat_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    if st.button("Load Chat Model"):
        if chat_model_name != st.session_state.current_model_name:
            clear_chat_model(chat_model, st.session_state)
            try:
                chat_model_config = AutoConfig.from_pretrained(chat_model_name)
                with st.spinner(f"Trying to load {chat_model_name}..."):
                    chat_model = load_model(chat_model_name, 
                                            vram_limit=0.55, 
                                            max_length=chat_model_config.max_position_embeddings, 
                                            use_lora=False)
                    st.session_state.chat_model = chat_model
                st.session_state.current_model_name = chat_model_name
                st.success(f"Successfully loaded new model: {chat_model_name}")
            except Exception as e:
                st.error(f"Failed to load {chat_model_name}")
                st.error(e)
        else:
            st.info(f"{chat_model_name} already loaded.")

    if st.session_state.chat_model:
        st.write(f"Current model: {st.session_state.current_model_name}")

    if ('messages' not in st.session_state) or clear_chat:
        st.session_state.messages = []
    if ('trans_messages' not in st.session_state) or clear_chat:
        st.session_state.trans_messages = []

    if not trans_to_english:
        messages = st.session_state.messages
        chat_instruction = 'Start a conversation in English.'
    else:
        messages = st.session_state.trans_messages
        chat_instruction = 'Start a conversation in 한국어 / 日本語 / 中文.'
    
    for msg in messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # if prompt := st.chat_input(chat_instruction):
        


if __name__ == '__main__':
    main()
