import re
import os
import sys
import json
from time import sleep

import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import OpenAI

sys.path.append('../../inference/codes')
from api_secret import OPENAI_CLIENT_KEY_TMAXNLP


LANG_TABLE = {
    'en': 'English',
    'ko': '한국어',
    'ja': '日本語',
    'zh': '中文',
}


class GptGenerator:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def get_gpt_system_prompt(self, src_lang_code='en', tgt_lang_code='ko'):
        src_lang_full = LANG_TABLE[src_lang_code]
        tgt_lang_full = LANG_TABLE[tgt_lang_code]
        gpt_system_instruction = f"""
            You are an assistant for making {src_lang_full}-{tgt_lang_full} text-pair data, for the another translation model.
            You have to modify the given text-pair, by following the task guidelines below.
        """
        gpt_system_instruction = re.sub(r'\s{2,}', '\n', gpt_system_instruction).strip()
        gpt_system_prompt = f"<instruction>\n{gpt_system_instruction}\n</instruction>"
        return gpt_system_prompt

    def generate(self, prompt, gpt_version='gpt-4o-mini', seed=42, src_lang_code='en', tgt_lang_code='ko'):
        system_prompt = self.get_gpt_system_prompt(src_lang_code, tgt_lang_code)
        
        response = self.client.chat.completions.create(
            model=gpt_version,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=1.0,
            seed=seed,
        )
        output = response.choices[0].message.content
        return output


def get_user_prompt_linebreak(src_text, tgt_text, src_lang_code='en', tgt_lang_code='ko', n=1):
    src_lang_full, tgt_lang_full = LANG_TABLE[src_lang_code], LANG_TABLE[tgt_lang_code]

    task_instruction = f"""
        Add {n} line-break(s) in each text in the pair.
        The {n} line-break(s) can be located anywhere, even in the middle of the sentence.
        But, the position of the line-breaks in two texts in the pair should be equivalent, regarding the semantic and syntactic structure of them.
        Which means, the number of the line-breaks should be the same in the two texts, and the number of broken sentence should be {n+1}.
        Do not add XML tags or any other special characters, just follow the output template.
        Use the examples as a guide and follow the instructions.
    """
    task_instruction = re.sub(r'\s{2,}', '\n', task_instruction).strip()
    task_prompt = f"<task>\n{task_instruction}\n</task>"

    output_instruction = f"""
        [SOURCE]\n{{src_text}}\n[/SOURCE]
        [TARGET]\n{{tgt_text}}\n[/TARGET]
    """
    output_instruction = re.sub(r'\s{2,}', '\n', output_instruction).strip()
    output_prompt = f"<output_template>\n{output_instruction}\n</output_template>"

    if n == 1:
        src_example = "안녕하세요. 제 이름은\nGPT-4o입니다."
        tgt_example = "Hello. My name is\nGPT-4o."
    elif n == 2:
        src_example = "안녕하세요.\n제 이름은\nGPT-4o입니다."
        tgt_example = "Hello.\nMy name is\nGPT-4o."
    elif n == 3:
        src_example = "안녕하세요.\n제\n이름은\nGPT-4o입니다."
        tgt_example = "Hello.\nMy\nname is\nGPT-4o."

    example_instruction = f"""
        <source><한국어>\n안녕하세요. 제 이름은 GPT-4o입니다.\n</한국어></source>
        <target><English>\nHello. My name is GPT-4o.\n</English></target>

        [SOURCE]\n{src_example}\n[/SOURCE]
        [TARGET]\n{tgt_example}\n[/TARGET]
    """
    example_instruction = re.sub(r'\s{2,}', '\n', example_instruction).strip()
    example_prompt = f"<example>\n{example_instruction}\n</example>"

    source_prompt = f"<source><{src_lang_full}>\n{src_text}\n</{src_lang_full}></source>"
    target_prompt = f"<target><{tgt_lang_full}>\n{tgt_text}\n</{tgt_lang_full}></target>"

    user_prompt = f"{task_prompt}\n\n{output_prompt}\n\n{example_prompt}\n\n{source_prompt}\n{target_prompt}"
    
    return user_prompt


def get_user_prompt_linebreak_modification(src_text, tgt_text, src_lang_code='en', tgt_lang_code='ko', n=1):
    src_lang_full, tgt_lang_full = LANG_TABLE[src_lang_code], LANG_TABLE[tgt_lang_code]

    task_instruction = f"""
        Two sentences will be given, a source sentence in {src_lang_full} and a target sentence in {tgt_lang_full}.
        Each sentence has a few line-breaks, but their locations are maybe not appropriate.
        There is no answer, but you can move the line-breaks' location to make the sentence more readable.

        The modification should satisfy the following conditions:
        (1) Each broken words should be semantically organized, not separated to the next line.
        (2) Each line-breaks' locations in the two sentences should be semantically/grammatically equivalent, if possible.
        (3) The number of the line-breaks can be changed, but it should not be 0 (no line-break).
        (4) If the locations are already appropriate, leave them as they are.
        (5) Never add/remove/change any words, just move the line-breaks.

        Remind the conditions, and modify the line-breaks in the sentences.
    """
    task_instruction = re.sub(r'\s{2,}', '\n', task_instruction).strip()
    task_prompt = f"<task>\n{task_instruction}\n</task>"

    output_instruction = f"""
        [SOURCE]\n{{source text with modified line-breaks}}\n[/SOURCE]
        [TARGET]\n{{target text with modified line-breaks}}\n[/TARGET]
    """
    output_instruction = re.sub(r'\s{2,}', '\n', output_instruction).strip()
    output_prompt = f"<output_template>\n{output_instruction}\n</output_template>"

    in_src_example = "I am aware that I am desperate\nfor each game and that it's\nnot natural."
    in_tgt_example = "저에게는 한 경기\n한 경기가 간절하고 당연한 게 아니라는 걸 잘\n알고 있다."
    out_src_example = "I am aware that I am desperate for each game\nand that it's not natural."
    out_tgt_example = "저에게는 한 경기 한 경기가 간절하고\n당연한 게 아니라는 걸 잘 알고 있다."

    example_instruction = f"""
        <source><English>\n{in_src_example}\n</English></source>
        <target><한국어>\n{in_tgt_example}\n</한국어></target>

        [SOURCE]\n{out_src_example}\n[/SOURCE]
        [TARGET]\n{out_tgt_example}\n[/TARGET]
    """
    example_instruction = re.sub(r'\s{2,}', '\n', example_instruction).strip()
    example_prompt = f"<example>\n{example_instruction}\n</example>"

    source_prompt = f"<source><{src_lang_full}>\n{src_text}\n</{src_lang_full}></source>"
    target_prompt = f"<target><{tgt_lang_full}>\n{tgt_text}\n</{tgt_lang_full}></target>"

    user_prompt = f"{task_prompt}\n\n{output_prompt}\n\n{example_prompt}\n\n{source_prompt}\n{target_prompt}"
    
    return user_prompt


def get_user_prompt_propernoun_contrastive(src_text, tgt_text, src_lang_code='en', tgt_lang_code='ko'):
    src_lang_full, tgt_lang_full = LANG_TABLE[src_lang_code], LANG_TABLE[tgt_lang_code]

    task_instruction = f"""
        Two types of texts will be given, a source text in {src_lang_full} and a target text in {tgt_lang_full}.
        You have to generate two types of modified target text:
        (1) Maintained: Translated into {tgt_lang_full}, with {src_lang_full} proper nouns.
        (2) Translated: Translated into {tgt_lang_full}, with {tgt_lang_full} proper nouns.

        Do not change any other parts of the text, except for the proper nouns.
        Do not add XML tags or any other special characters, just follow the output template.
        Use the examples as a guide and follow the instructions.

        If there is no proper noun in the text, the output should be "N/A" within the template.
    """
    task_instruction = re.sub(r'\s{2,}', '\n', task_instruction).strip()
    task_prompt = f"<task>\n{task_instruction}\n</task>"

    output_instruction = f"""
        [PN_MAINTAINED]\n{{{tgt_lang_full} target text with {src_lang_full} proper nouns, or N/A}}\n[/PN_MAINTAINED]
        [PN_TRANSLATED]\n{{{tgt_lang_full} target text with {tgt_lang_full} proper nouns, or N/A}}\n[/PN_TRANSLATED]
    """
    output_instruction = re.sub(r'\s{2,}', '\n', output_instruction).strip()
    output_prompt = f"<output_template>\n{output_instruction}\n</output_template>"

    example_instruction_input = f"""
        <source><English>\nThe Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.\n</English></source>
        <target><한국어>\n에펠탑은 프랑스 파리 Champ de Mars에 있는 철제 격자 탑입니다.\n</한국어></target>
    """
    example_instruction_input = re.sub(r'\s{2,}', '\n', example_instruction_input).strip()
    example_instruction_output = f"""
        [PN_MAINTAINED]\nEiffel Tower는 France Paris의 Champ de Mars에 있는 철제 격자 탑입니다.\n[/PN_MAINTAINED]
        [PN_TRANSLATED]\n에펠탑은 프랑스 파리의 샹드마르에 있는 철제 격자 탑입니다.\n[/PN_TRANSLATED]
    """
    example_instruction_output = re.sub(r'\s{2,}', '\n', example_instruction_output).strip()
    example_prompt = f"<example>\n{example_instruction_input}\n\n{example_instruction_output}\n</example>"

    source_prompt = f"<source><{src_lang_full}>\n{src_text}\n</{src_lang_full}></source>"
    target_prompt = f"<target><{tgt_lang_full}>\n{tgt_text}\n</{tgt_lang_full}></target>"

    user_prompt = f"{task_prompt}\n\n{output_prompt}\n\n{example_prompt}\n\n{source_prompt}\n{target_prompt}"
    user_prompt = user_prompt.strip()
    
    return user_prompt


def get_user_prompt_style_contrastive(src_text, tgt_text, src_lang_code='en', tgt_lang_code='ko'):
    src_lang_full, tgt_lang_full = LANG_TABLE[src_lang_code], LANG_TABLE[tgt_lang_code]

    task_instruction = f"""
        Two types of texts will be given, a source text in {src_lang_full} and a target text in {tgt_lang_full}.
        You have to generate two types of outputs:
        (1) Translated into {tgt_lang_full}, in formal and polite (written-alike) style.
        (2) Translated into {tgt_lang_full}, in informal and casual (colloquial-alike) style.
        
        The structure and vocabulary of the target text can change significantly based on the specified style.
        Do not add XML tags or any other special characters, just follow the output template.
        Use the examples as a guide and follow the instructions.
    """
    task_instruction = re.sub(r'\s{2,}', '\n', task_instruction).strip()
    task_prompt = f"<task>\n{task_instruction}\n</task>"

    output_instruction = f"""
        [FORMAL]\n{{{tgt_lang_full} translation in formal style}}\n[/FORMAL]
        [INFORMAL]\n{{{tgt_lang_full} translation in informal style}}\n[/INFORMAL]
    """
    output_instruction = re.sub(r'\s{2,}', '\n', output_instruction).strip()
    output_prompt = f"<output_template>\n{output_instruction}\n</output_template>"

    example_instruction_input = f"""
        <source><English>\nThe authorities have announced their plan to distribute free masks to individuals who arrive for their COVID-19 vaccination this weekend.\n</English></source>
        <target><한국어>\n정부는 이번 주말에 코로나 백신 접종을 받으러 오는 사람들에게 무료로 마스크를 제공할 예정이라고 밝혔다.\n</한국어></target>
    """
    example_instruction_input = re.sub(r'\s{2,}', '\n', example_instruction_input).strip()
    example_instruction_output = f"""
        [FORMAL]\n당국은 금주 주말 코로나 백신 접종을 위해 방문하는 이들에게 무료 마스크를 제공할 예정이라 공지했다.\n[/FORMAL]
        [INFORMAL]\n정부에서 이번 주말에 코로나 백신 맞으러 오는 사람들한테 공짜로 마스크 나눠준다고 했어요.\n[/INFORMAL]
    """
    example_instruction_output = re.sub(r'\s{2,}', '\n', example_instruction_output).strip()
    example_prompt = f"<example>\n{example_instruction_input}\n\n{example_instruction_output}\n</example>"

    source_prompt = f"<source><{src_lang_full}>\n{src_text}\n</{src_lang_full}></source>"
    target_prompt = f"<target><{tgt_lang_full}>\n{tgt_text}\n</{tgt_lang_full}></target>"

    user_prompt = f"{task_prompt}\n\n{output_prompt}\n\n{example_prompt}\n\n{source_prompt}\n{target_prompt}"

    return user_prompt


def generate_single(src_text, tgt_text, prompt_type, src_lang_code='en', tgt_lang_code='ko', **kwargs):
    if prompt_type == 'linebreak':
        num_linebreaks = kwargs.get('n', 1)
        user_prompt = get_user_prompt_linebreak(src_text, tgt_text, src_lang_code, tgt_lang_code, n=num_linebreaks)
    elif prompt_type == 'propernoun_contrastive':
        user_prompt = get_user_prompt_propernoun_contrastive(src_text, tgt_text, src_lang_code, tgt_lang_code)
    elif prompt_type == 'style_contrastive':
        user_prompt = get_user_prompt_style_contrastive(src_text, tgt_text, src_lang_code, tgt_lang_code)
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")

    gpt_generator = GptGenerator(api_key=OPENAI_CLIENT_KEY_TMAXNLP)
    output = gpt_generator.generate(user_prompt)

    return output


def make_jsonl_request(df, prompt_type, jsonl_path, gpt_version='gpt-4o-mini', **kwargs):
    user_prompt_func = {
        # 'linebreak': get_user_prompt_linebreak,
        # 'propernoun': get_user_prompt_propernoun_contrastive,
        # 'style': get_user_prompt_style_contrastive,
        'linebreak-modification': get_user_prompt_linebreak_modification,
    }
    
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r') as f:
            request_list = [json.loads(line) for line in f]
    else:
        request_list = []

    generator = GptGenerator(api_key=OPENAI_CLIENT_KEY_TMAXNLP)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Making JSONL Requests"):
        src_text, tgt_text = row['src'], row['tgt']
        src_lang, tgt_lang = row['direction'].split('-')

        system_prompt = generator.get_gpt_system_prompt(src_lang, tgt_lang)

        user_prompt = user_prompt_func[prompt_type](src_text, tgt_text, src_lang, tgt_lang, **kwargs)
        request = {
            "custom_id": f"{prompt_type}-{src_lang}{tgt_lang}-{idx+1}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": gpt_version,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    'temperature': 1.0,
                    'seed': 42
                }
        }
        request_list.append(request)

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for request in request_list:
            f.write(json.dumps(request, ensure_ascii=False) + '\n')
    

def reverse_direction(df):
    df_reverse = df.copy()
    df_reverse['src'], df_reverse['tgt'] = df['tgt'], df['src']
    df_reverse['src_xcomet'], df_reverse['tgt_xcomet'] = df['tgt_xcomet'], df['src_xcomet']
    df_reverse['direction'] = df['direction'].apply(lambda x: '-'.join(x.split('-')[::-1]))
    return df_reverse


def upload_jsonl_request(client, df, prompt_type, jsonl_path, gpt_version, **kwargs): 
    make_jsonl_request(df, prompt_type, jsonl_path, gpt_version, **kwargs)

    client.files.create(
        file=open(jsonl_path, 'rb'),
        purpose='batch'
    )
    file_info = list(client.files.list())[0]
    print(f"Uploaded JSONL file: {jsonl_path} | File ID: {file_info.id}")

    return file_info


def inference_batch(client, file_id):
    client.batches.create(
        input_file_id=file_id,
        endpoint='/v1/chat/completions',
        completion_window='24h',
    )
    batch_info = list(client.batches.list())[0]
    print(f"Started inference batch: {batch_info.id} | File ID: {file_id}")

    return batch_info


def check_batch_status(num=-1):
    client = OpenAI(api_key=OPENAI_CLIENT_KEY_TMAXNLP)
    batches = client.batches.list()
    for batch in list(batches)[:num]:
        batch_id = batch.id
        batch_info = client.batches.retrieve(batch_id=batch_id)
        print("############################################")
        print(f"Batch ID: {batch_id}")
        print(f"Status: {batch_info.status}")
        if batch_info.status == 'failed':
            print(f"Error: {list(batch_info.errors)[0]}")
        print(f"Progress: {batch_info.request_counts.completed}/{batch_info.request_counts.total} ({batch_info.request_counts.failed} failed)")
        print("############################################")


def check_files(num=-1):
    client = OpenAI(api_key=OPENAI_CLIENT_KEY_TMAXNLP)
    files = client.files.list()
    for file in list(files)[:num]:
        print(f"ID: {file.id} | Local Name: {file.filename} | Purpose: {file.purpose}")


def check_batches(num=-1):
    client = OpenAI(api_key=OPENAI_CLIENT_KEY_TMAXNLP)
    batches = client.batches.list()
    for batch in list(batches)[:num]:
        print(f"ID: {batch.id} | Status: {batch.status} | Progress: {batch.request_counts.completed}/{batch.request_counts.total} ({batch.request_counts.failed} failed)")
        print(f"Input File ID: {batch.input_file_id} | Output File ID: {batch.output_file_id}")


def make_upload_inference(gpt_version='gpt-4o-mini'):
    client = OpenAI(api_key=OPENAI_CLIENT_KEY_TMAXNLP)

    # df_path = '../../inference/results/prime-cleansing/sft/mmt_not_prime_train.csv'
    df_path = './dpo_unpreprocessed.csv'
    df = pd.read_csv(df_path)

    gen_dict = {
        # 'linebreak': {'size': 650, 'jsonl_path': './gpt_dpo_linebreak_request.jsonl'},
        # 'propernoun': {'size': 2200, 'jsonl_path': './gpt_dpo_propernoun_request.jsonl'},
        # 'style': {'size': 1300, 'jsonl_path': './gpt_dpo_style_request_koen.jsonl'},
        'linebreak-modification': {'size': 650, 'jsonl_path': './gpt_dpo_linebreak_modification_request.jsonl'},
    }
    file_infos = {}
    for prompt_type, params in gen_dict.items():
        df_size = params['size']
        jsonl_path = params['jsonl_path']
        df_batch = df[df['dpo_info'] == 'lb-reflected']
        # df_batch = pd.DataFrame()
        # for lang in ['en', 'ja', 'zh']:
        #     df_lang = df[df['direction'].str.contains(lang)].sample(n=df_size * 2, random_state=42)
        #     df_batch_src2tgt = df_lang.iloc[:df_size]
        #     df_batch_tgt2src = reverse_direction(df_lang.iloc[df_size:])
        #     df_batch_subset = pd.concat([df_batch_src2tgt, df_batch_tgt2src], ignore_index=True)
        #     df_batch = pd.concat([df_batch, df_batch_subset], ignore_index=True)
        kwargs = {'n': 1} if prompt_type == 'linebreak' else {}
        
        file_info = upload_jsonl_request(client, df_batch, prompt_type, jsonl_path, gpt_version, **kwargs)
        file_infos[prompt_type] = file_info
    
    batch_infos = {}
    for prompt_type, file_info in file_infos.items():
        batch_info = inference_batch(client, file_info.id)
        for prompt_type, info in file_infos.items():
            if info.id == batch_info.input_file_id:
                batch_infos[prompt_type] = batch_info


def download_responses(file_id, output_path):
    client = OpenAI(api_key=OPENAI_CLIENT_KEY_TMAXNLP)
    response = client.files.content(file_id)
    
    lines = response.text.strip().split('\n')
    json_objects = [json.loads(line) for line in lines]

    with open(output_path, 'w', encoding='utf-8') as f:
        for obj in json_objects:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    # # Batch inference
    # make_upload_inference('gpt-4o')
    # sleep(10)
    # check_batches(1)

    # # Check batch status
    # check_batches(2)

    # Download Batch
    download_responses('file-LkH1guWU1m6Cf1mcfxk6Cl7O', './gpt_dpo_linebreak_modification_response.jsonl')

    # client = OpenAI(api_key=OPENAI_CLIENT_KEY_TMAXNLP)

    # # Cancel batch
    # batch_num = 0
    # batch_id = list(client.batches.list())[batch_num].id
    # client.batches.cancel(batch_id)
