import sys

import torch
from openai import OpenAI
from urllib import parse, request
from deepl import Translator as DeeplTranslator
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

sys.path.append('./')
from api_secret import (
    PAPAGO_CLIENT_ID,
    PAPAGO_CLIENT_SECRET,
)
from api_secret import (
    DEEPL_CLIENT_KEY,
)
from api_secret import (
    OPENAI_CLIENT_KEY,
)
from inference.codes.translation_info import *


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PapagoTranslator:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.url = 'https://naveropenapi.apigw.ntruss.com/nmt/v1/translation'

    def translate(self, text, src_lang='en', tgt_lang='ko'):
        encoded_text = parse.quote(text)
        data = f'source={src_lang}&target={tgt_lang}&text={encoded_text}'
        
        trans_request = request.Request(self.url)
        trans_request.add_header('X-NCP-APIGW-API-KEY-ID', self.client_id)
        trans_request.add_header('X-NCP-APIGW-API-KEY', self.client_secret)

        trans_response = request.urlopen(trans_request, data=data.encode('utf-8'))
        responded_code = trans_response.getcode()

        if responded_code == 200:
            responded_body = trans_response.read()
            translation = responded_body.decode('utf-8')
            translation = eval(translation)['message']['result']['translatedText']
            return translation
        else:
            raise Exception(f"HTTPError: {responded_code}")


class GptTranslator:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def translate(self, text, src_lang='ko', tgt_lang='en', gpt_version='gpt-4o-mini', prompt_type='llama'):
        GPT_SYSTEM_PROMPT_STRIPPED = GPT_SYSTEM_PROMPT.strip().replace("\n", " ")
        user_prompt = make_prompt(text, src_lang, tgt_lang, prompt_type)
        
        response = self.client.chat.completions.create(
            model=gpt_version,
            messages=[
                {"role": "system", "content": GPT_SYSTEM_PROMPT_STRIPPED},
                {"role": "user", "content": user_prompt}
            ]
        )
        translation = response.choices[0].message.content
        return translation


class ApiTranslator:
    def __init__(self, model_type):
        self.model_type = model_type
        self.model = None

    def load_model(self):
        if self.model_type == 'papago':
            return PapagoTranslator(PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET)
        elif self.model_type == 'deepl':
            return DeeplTranslator(DEEPL_CLIENT_KEY)
        elif self.model_type == 'gpt':
            return GptTranslator(api_key=OPENAI_CLIENT_KEY)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    def translate(self, texts, src_lang, tgt_lang, **kwargs):
        translations = []
        for text, src, tgt in zip(texts, src_lang, tgt_lang):
            if self.model_type == 'papago':
                translation = self.model.translate(text, src_lang=PAPAGO_LANG_CODE[src], tgt_lang=PAPAGO_LANG_CODE[tgt])
            elif self.model_type == 'deepl':
                translation = self.model.translate_text(text, source_lang=DEEPL_LANG_CODE[src]['src'], target_lang=DEEPL_LANG_CODE[tgt]['tgt'])
            elif self.model_type == 'gpt':
                translation = self.model.translate(text, src_lang=src, tgt_lang=tgt, **kwargs)
            else:
                raise ValueError(f"Invalid model type: {self.model_type}")
            translations.append(translation)
        
        return translations
        

class HfTranslator:
    def __init__(self, model_path, max_length):
        self.model_path = model_path
        self.max_length = max_length
        self.model = None
        self.tokenizer = None

    def load_model(self, lora_path=None, adapter_name=None, quantization=None, torch_dtype=torch.bfloat16, cache_dir=HF_CACHE_DIR):
        if quantization == 'nf4':
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
        else:
            quantization_config = None
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                max_length=self.max_length,
                quantization_config=quantization_config,
                attn_implementation='flash_attention_2',
                torch_dtype=torch_dtype, 
                cache_dir=cache_dir,
            )
        except:
            print(f"Model {self.model_path} is not a CausalLM, trying Seq2SeqLM...")
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_path, 
                    max_length=self.max_length,
                    quantization_config=quantization_config,
                    torch_dtype=torch_dtype, 
                    cache_dir=cache_dir,
                )
            except:
                raise ValueError(f"Model {self.model_path} is not a CausalLM or Seq2SeqLM neither. Try again with a proper model.")
        
        if quantization == None:
            model.to(DEVICE)

        if lora_path is not None:
            model = PeftModel.from_pretrained(
                model,
                lora_path,
                adapter_name=adapter_name,
                torch_dtype=torch_dtype,
            )

        return model

    def load_tokenizer(self, padding_side='left'):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        tokenizer.padding_side = padding_side
        tokenizer.model_max_length = self.max_length
        if 'llama' in self.model_path.lower():
            if any(model_suffix in self.model_path.lower() for model_suffix in ['llama-2', 'llama2']):
                tokenizer.pad_token_id = 2
                tokenizer.eos_token_id = 46332
            elif any(model_suffix in self.model_path.lower() for model_suffix in ['llama-3', 'llama3']):
                tokenizer.pad_token_id = 128002
        elif 'mbart' in self.model_path:
            tokenizer.src_lang = MBART_LANG_CODE[self.src_lang]
            tokenizer.tgt_lang = MBART_LANG_CODE[self.tgt_lang]
        try:
            tokenizer.add_eos_token = True
        except:
            print(f"Tokenizer {self.model_path} does not support 'add_eos_token'.")
        
        return tokenizer

    def translate(self, prompts):
        inputs = self.tokenizer(
            prompts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=self.max_length
        )
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

        for idx, (input_ids, attn_mask) in enumerate(zip(inputs['input_ids'], inputs['attention_mask'])):
            if input_ids[-1] != self.tokenizer.eos_token_id:
                continue
            inputs['input_ids'][idx] = input_ids[:-1]
            inputs['attention_mask'][idx] = attn_mask[:-1]

        outputs = self.model.generate(
            **inputs, 
            max_length=self.max_length, 
            eos_token_id=self.tokenizer.eos_token_id
        )
        outputs = [outputs[idx][inputs['input_ids'][idx].shape[0]:] for idx in range(len(outputs))]
        translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations = [translation.strip() for translation in translations]
        
        return translations
    

class VllmTranslator:
    def __init__(self, model_path, lora_path=None, adapter_name=None):
        self.model_path = model_path
        self.lora_path = lora_path
        self.adapter_name = adapter_name
        self.model = None
        self.sampling_params = None

    def load_model(self,
                   max_length=MAX_LENGTH,
                   max_lora_rank=VLLM_MAX_LORA_RANK,
                   seed=SEED,
                   torch_dtype=torch.bfloat16,
                   cache_dir=HF_CACHE_DIR,
                   vram_limit=VLLM_VRAM_LIMIT,
                   tensor_parallel_size=TENSOR_PARALLEL_SIZE):
        model = LLM(
            model=self.model_path,
            tokenizer=self.model_path,
            max_seq_len_to_capture=max_length,
            enable_lora=True if self.lora_path is not None else False,
            max_lora_rank=max_lora_rank if self.lora_path is not None else None,
            seed=seed,
            dtype=torch_dtype,
            trust_remote_code=True,
            download_dir=cache_dir,
            gpu_memory_utilization=vram_limit,
            tensor_parallel_size=tensor_parallel_size,
        )
        return model

    def load_sampling_params(self, 
                             temperature=0.0, 
                             use_beam_search=False, 
                             best_of=1,
                             top_k=40, 
                             top_p=0.95, 
                             skip_special_tokens=True, 
                             stop=None, 
                             repetition_penalty=1.1, 
                             max_tokens=MAX_LENGTH):
        sampling_params = SamplingParams(
            temperature=temperature,
            use_beam_search=use_beam_search,
            best_of=best_of,
            top_k=top_k,
            top_p=top_p,
            skip_special_tokens=skip_special_tokens,
            stop=stop,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
        )
        return sampling_params

    def translate(self, prompts):
        lora_request = LoRARequest(self.adapter_name, 1, self.lora_path) if self.lora_path is not None else None
        outputs = self.model.generate(prompts, self.sampling_params, lora_request=lora_request, use_tqdm=False)
        translations = [output.outputs[0].text.strip() for output in outputs]
        return translations
        

def make_prompt(text, src_lang, tgt_lang, prompt_type=None):
    if tgt_lang is None:
        raise ValueError("tgt_lang must be provided.")
    
    if prompt_type is None:
        prompt = text
    elif prompt_type == 'llama':
        if src_lang is None:
            raise ValueError("src_lang must be provided.")
        instruction = f"Translate this from {LLAMA_LANG_TABLE[src_lang]} to {LLAMA_LANG_TABLE[tgt_lang]}."
        src_suffix = f"### {LLAMA_LANG_TABLE[src_lang]}:"
        tgt_suffix = f"### {LLAMA_LANG_TABLE[tgt_lang]}:"
        prompt = f"{instruction}\n{src_suffix} {text}\n{tgt_suffix}"
    elif prompt_type == 'madlad':
        src_suffix = f"<2{MADLAD_LANG_CODE[tgt_lang]}>"
        prompt = f"{src_suffix} {text}"
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")
    
    return prompt
