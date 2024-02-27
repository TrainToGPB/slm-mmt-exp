"""
Perform inference on the specified dataset using the specified pre-trained language model.

The following models are supported:
- opus (제외: 번역 안됨)
- mbart
- nllb-600m
- nllb-1.3b (제외)
- madlad
- mbart-aihub
- llama
- llama-aihub-qlora
- llama-aihub-qlora-bf16 (merged & upscaled)
- llama-aihub-qlora-fp16 (merged & upscaled)
- llama-aihub-qlora-bf16-vllm (merged & upscaled + vLLM)
- llama-aihub-qlora-augment (확장된 데이터)
- llama-aihub-qlora-reverse-new (llama-aihub-qlora 체크포인트에서 새로운 데이터로 한-영 역방향 학습)
- llama-aihub-qlora-reverse-overlap (llama-aihub-qlora 체크포인트에서 동일한 데이터로 한-영 역방향 학습)

The following datasets are supported:
- aihub: AI Hub integrated dataset (ref: https://huggingface.co/datasets/traintogpb/aihub-koen-translation-integrated-tiny-100k)
- flores: FLoRes-101 dataset (ref: https://huggingface.co/datasets/gsarti/flores_101)

CLI example:
- Inference on the AI Hub dataset:
    $ python model_inferences.py --model_type=mbart --inference_type=dataset --dataset=aihub
- Inference on a single sentence:
    $ python model_inferences.py --model_type=llama-bf16-vllm --inference_type=sentence --sentence="Hello, world!"

Output:
- Translated dataset file (CSV format)
- Translated sentence (print)

Notes:
- The translated dataset file will be saved in the same directory as the original dataset file.
- The translated sentence will be printed on the console.
"""
# built-in
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['VLLM_USE_MODELSCOPE'] = 'false'
import re
import sys
import argparse
from tqdm import tqdm

# third-party
import torch
import pandas as pd
from peft import PeftModel
from vllm import LLM, SamplingParams
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import MarianMTModel, MarianTokenizer
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from transformers import M2M100ForConditionalGeneration, NllbTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from transformers import BitsAndBytesConfig

# custom
sys.path.append('./')
sys.path.append('../../')
from training.training_utils import set_seed


SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)


def load_model_and_tokenizer(model_type):
    """
    Load pre-trained language model and tokenizer based on the model type.

    Parameters:
    - model_type (str): Type of the pre-trained language model.

    Returns:
    - model (PreTrainedModel): Pre-trained language model.
    - tokenizer (PreTrainedTokenizer): Pre-trained tokenizer.
    """
    # Model mapping
    model_mapping = {
        'opus': ('Helsinki-NLP/opus-mt-tc-big-en-ko', MarianMTModel, MarianTokenizer),
        'mbart': ('facebook/mbart-large-50-many-to-many-mmt', MBartForConditionalGeneration, MBart50Tokenizer),
        'nllb-600m': ('facebook/nllb-200-distilled-600M', M2M100ForConditionalGeneration, NllbTokenizer),
        'nllb-1.3b': ('facebook/nllb-200-distilled-1.3B', M2M100ForConditionalGeneration, NllbTokenizer),
        'madlad': ('google/madlad400-3b-mt', T5ForConditionalGeneration, T5Tokenizer),
        'mbart-aihub': (os.path.join(SCRIPT_DIR, '../../training/mbart/models/mbart-full'), MBartForConditionalGeneration, MBart50Tokenizer),
        'llama': ('beomi/open-llama-2-ko-7b', LlamaForCausalLM, LlamaTokenizer),
        'llama-aihub-qlora': (('beomi/open-llama-2-ko-7b', 'traintogpb/llama-2-en2ko-translator-7b-qlora-adapter'), LlamaForCausalLM, LlamaTokenizer),
        'llama-aihub-qlora-bf16': ('traintogpb/llama-2-en2ko-translator-7b-qlora-bf16-upscaled', LlamaForCausalLM, LlamaTokenizer),
        'llama-aihub-qlora-fp16': (os.path.join(SCRIPT_DIR, '../../training/llama_qlora/models/baseline-merged-fp16'), LlamaForCausalLM, LlamaTokenizer),
        'llama-aihub-qlora-bf16-vllm': ('traintogpb/llama-2-en2ko-translator-7b-qlora-bf16-upscaled', None, LlamaTokenizer),
        'llama-aihub-qlora-augment': (('beomi/open-llama-2-ko-7b', os.path.join(SCRIPT_DIR, '../../training/llama_qlora/models/augment')), LlamaForCausalLM, LlamaTokenizer),
        'llama-aihub-qlora-reverse-new': (('beomi/open-llama-2-ko-7b', os.path.join(SCRIPT_DIR, '../../training/llama_qlora/models/continuous-reverse-new')), LlamaForCausalLM, LlamaTokenizer),
        'llama-aihub-qlora-reverse-overlap': (('beomi/open-llama-2-ko-7b', os.path.join(SCRIPT_DIR, '../../training/llama_qlora/models/continuous-reverse-overlap')), LlamaForCausalLM, LlamaTokenizer),
        'mt5-aihub-base-fft': (os.path.join(SCRIPT_DIR, '../../training/mt5/models/base-fft-en2ko-separate-token-constlr'), MT5ForConditionalGeneration, T5Tokenizer),
    }
    assert model_type in model_mapping.keys(), 'Wrong model type'

    # Load pre-trained language model and tokenizer
    model_name, model_cls, tokenizer_cls = model_mapping[model_type]
    if isinstance(model_name, tuple):
        model_name, adapter_path = model_name[0], model_name[1]
    
    # llama-aihub-qlora, llama-aihub-qlora-bf16, llama-aihub-qlora-fp16
    if model_type.startswith('llama-aihub-qlora'):
        if '16' in model_type:
            if model_type.endswith('vllm'):                # bf16-vllm
                model = LLM(model=model_name, seed=SEED)
            else:
                if model_type.endswith('bf16'):
                    model = model_cls.from_pretrained(model_name, torch_dtype=torch.bfloat16)
                elif model_type.endswith('fp16'):
                    model = model_cls.from_pretrained(model_name, torch_dtype=torch.float16)
        else:                                              # baseline, augment, reverse-new, reverse-overlap
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
            torch_dtype = torch.bfloat16
            model = model_cls.from_pretrained(
                model_name, 
                quantization_config=bnb_config, 
                torch_dtype=torch_dtype
            )
            model = PeftModel.from_pretrained(
                model, 
                adapter_path, 
                torch_dtype=torch_dtype
            )

        tokenizer = tokenizer_cls.from_pretrained(model_name)
        tokenizer.pad_token = "</s>"
        tokenizer.pad_token_id = 2
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.eos_token_id = 46332
        tokenizer.add_eos_token = True
        tokenizer.padding_side = 'right'
        tokenizer.model_max_length = 768

    # opus, mbart, nllb-600m, nllb-1.3b, madlad, mbart-aihub, llama
    else:
        model = model_cls.from_pretrained(model_name)
        tokenizer = tokenizer_cls.from_pretrained(model_name)

    return model, tokenizer


def translate(model, tokenizer, text, model_type, print_result=False, max_length=512):
    """
    Translate the input text using the pre-trained language model.

    Parameters:
    - model (PreTrainedModel): Pre-trained language model.
    - tokenizer (PreTrainedTokenizer): Pre-trained tokenizer.
    - text (str): Input text to be translated.
    - model_type (str): Type of the pre-trained language model.
    - print_result (bool): Whether to print the translated text.

    Returns:
    - translated_text (str): Translated text.
    """
    if not model_type.endswith('vllm'):
        model.to(DEVICE)

    if model_type == 'madlad':
        text = f"<2ko> {text}"
    elif 'llama' in model_type:
        # few_shot_dict = {
        #     "English": ["Now K-Pop has become a dream of many Brazilian teenagers, who are eager to have more stages and opportunities to fulfill it.",
        #                 "Aichi Prefecture, one of the regions where the state of emergency is being reviewed early, had 86 new cases the day before, and fell below 100 in four days.",
        #                 "The scenery of the town is really pretty, isn't it?",
        #                 "Elon Musk, the founder of the private space company SpaceX, promised a trip to Mars in 2020.",
        #                 "Would you tell us what actions make the advertisements pop up?",
        #                 "If you don't pay, you will suffer some losses when it comes to using your credit cards.",
        #                 "To efficiently examine agenda items, the Steering Committee may establish subcommittees for each field by resolution.",
        #                 "The human brain is known to have only two percent of total body weight, but to consume 20 percent of its total energy."],
        #     "한국어": ["이제 K-Pop은 많은 브라질 청소년들의 꿈이 되었고 그 꿈을 펼칠 수 있는 무대와 기회들이 더 많아지길 브라질 청소년들은 간절히 바라고 있다.",
        #                 "긴급사태 조기 해제가 검토되는 지역 중 하나인 아이치현은 전날 신규 확진자가 86명으로 4일 만에 100명 밑으로 떨어졌다.",
        #                 "마을의 풍경이 정말 예쁘죠?",
        #                 "민간 우주기업 스페이스엑스의 창업자 일론 머스크는 2020년 화성여행을 장담했다.",
        #                 "어떤 동작을 취했을 때 광고가 뜨는지 말씀해주시겠어요?",
        #                 "결제하지 않을 경우 카드 사용에 불이익이 있을 수 있습니다.",
        #                 "운영위원회는 안건을 효율적으로 심사하기 위하여 그 의결로 분야별 소위원회를 둘 수 있다.",
        #                 "인간의 뇌는 전체 체중의 2%에 지나지 않지만 전체 에너지의 20%를 소모한다고 알려져 있다."]
        # }
        # shot_num = 0
        text = f"### English: {text}\n### 한국어: "
    elif model_type.startswith('mt5'):
        text = f"<en> {text} <ko>"

    if 'mbart' in model_type:
        src_lang = 'en_XX'
        tgt_lang = 'ko_KR'
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang
    elif 'nllb' in model_type:
        src_lang = 'eng_Latn'
        tgt_lang = 'kor_Hang'
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang

    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    if model_type.endswith('vllm'):
        sampling_params = SamplingParams(
            temperature=0, 
            top_p=0.95,
            skip_special_tokens=True,
            stop='<|endoftext|>',
            repetition_penalty=1.0,
            max_tokens=350
        )
        outputs = model.generate([text], sampling_params, use_tqdm=False)
        translated_text = outputs[0].outputs[0].text
    else:
        if model_type.startswith('llama-aihub-qlora'):
            inputs['input_ids'] = inputs['input_ids'][0][:-1].unsqueeze(dim=0)
            inputs['attention_mask'] = inputs['attention_mask'][0][:-1].unsqueeze(dim=0)
            outputs = model.generate(**inputs, max_length=max_length, eos_token_id=46332)
        elif 'mbart' in model_type or 'nllb' in model_type:
            outputs = model.generate(**inputs, max_length=max_length, forced_bos_token_id=tokenizer.lang_code_to_id[tokenizer.tgt_lang])
        else:
            outputs = model.generate(**inputs, max_length=max_length)
        
        input_len = len(inputs['input_ids'].squeeze()) if model_type.startswith('llama') else 0
        
        translated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    translated_text = re.sub(r'\s+', ' ', translated_text)
    translated_text = translated_text.strip()
    
    if print_result:
        print(translated_text)

    return translated_text


def inference(model_type, source_column, target_column, file_path, print_result=False):
    """
    Perform inference on the specified dataset using the specified pre-trained language model.

    Parameters:
    - model_type (str): Type of the pre-trained language model.
    - source_column (str): Name of the source column in the dataset.
    - target_column (str): Name of the target column in the dataset.
    - file_path (str): Path to the dataset file.
    - print_result (bool): Whether to print the translated text.
    """
    set_seed(SEED)
    model, tokenizer = load_model_and_tokenizer(model_type)
    if not model_type.endswith('vllm'):
        model.to(DEVICE)

    max_length = 768 if 'llama' in model_type else 512

    eval_df = pd.read_csv(file_path)
    tqdm.pandas(desc="Translating")
    eval_df[target_column] = eval_df[source_column].progress_apply(lambda text: translate(model, tokenizer, text, model_type, print_result, max_length))
    eval_df.to_csv(file_path, index=False)


def inference_single(model_type, text):
    """
    Perform inference on a single sentence using the specified pre-trained language model.

    Parameters:
    - model_type (str): Type of the pre-trained language model.
    - text (str): Input text to be translated.

    Returns:
    - translation (str): Translated text.
    """
    set_seed(SEED)
    model, tokenizer = load_model_and_tokenizer(model_type)
    if not model_type.endswith('vllm'):
        model.to(DEVICE)

    translation = translate(model, tokenizer, text, model_type, max_length=768)

    return translation


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='llama-bf16', help="Pre-trained language model type for inference (e.g., llama-bf16)")
    parser.add_argument("--inference_type", type=str, default='sentence', help="Inference type (sentence or dataset)")
    parser.add_argument("--dataset", type=str, default='sample', help="Dataset path for inference (only for dataset inference, preset: aihub / flores / sample, or custom path to a CSV file)")
    parser.add_argument("--sentence", type=str, default="NMIXX is a South Korean girl group that made a comeback on January 15, 2024 with their new song 'DASH'.", help="Input English text for inference (only for sentence inference)")
    args = parser.parse_args()
    dataset = args.dataset
    
    source_column = "en"
    if args.dataset.endswith('.csv'):
        file_path = args.dataset
    else:
        file_path_dict = {
            'sample': os.path.join(SCRIPT_DIR, "../../sample_texts_for_inference.csv"),
            'aihub': os.path.join(SCRIPT_DIR, "../results/test_tiny_uniform100_inferenced.csv"),
            'flores': os.path.join(SCRIPT_DIR, "../results/test_flores_inferenced.csv")
        }
        file_path = file_path_dict[dataset]

    # model_type_candidates = [
    #     'mbart',
    #     'nllb-600m',
    #     'madlad',
    #     'llama',
    #     'mbart-aihub',
    #     'llama-aihub-qlora',
    #     'llama-aihub-qlora-bf16',
    #     'llama-aihub-qlora-fp16',
    #     'llama-aihub-qlora-bf16-vllm', # Best model
    #     'llama-aihub-qlora-augment',
    #     'llama-aihub-qlora-reverse-new',
    #     'llama-aihub-qlora-reverse-overlap'
    # ]
    model_type_dict = {
        'mbart': 'mbart',
        'nllb': 'nllb-600m',
        'madlad': 'madlad',
        'llama': 'llama-aihub-qlora',
        'llama-bf16': 'llama-aihub-qlora-bf16',
        'llama-bf16-vllm': 'llama-aihub-qlora-bf16-vllm',
        'mt5-fft': 'mt5-aihub-base-fft'
    }
    
    model_type = model_type_dict[args.model_type]
    print(f"Inference model: {model_type.upper()}")

    if args.inference_type == 'dataset':
        target_column = model_type + "-1shot_trans"
        inference(model_type, source_column, target_column, file_path, print_result=True)
    
    if args.inference_type == 'sentence':
        translation = inference_single(model_type, args.sentence)
        print(translation)
