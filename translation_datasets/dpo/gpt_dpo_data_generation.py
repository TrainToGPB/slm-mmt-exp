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
            You have to modify the given text-pair, by following the instructions below.
        """
        gpt_system_instruction = re.sub(r'\s{2,}', '\n', gpt_system_instruction).strip()
        gpt_system_prompt = f"<task>\n{gpt_system_instruction}\n</task>"
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


def get_user_prompt_linebreak(src_text, tgt_text, src_lang_code='en', tgt_lang_code='ko'):
    src_lang_full, tgt_lang_full = LANG_TABLE[src_lang_code], LANG_TABLE[tgt_lang_code]

    task_instruction = f"""
        Add line-break(s) in each text in the pair.
        
        1. The location of the line-break(s) can be anywhere in the text (i.e., end of a sentence, middle of a sentence).
        2. the line-break(s) should be inserted in a way that breaks small semantic units (e.g., phrases or clauses) within a sentence.
        3. The location of the line-breaks in each text should be equivalent, regarding the semantic and syntactic structure of them.
        4. The number of the line-breaks must be the same in each text.

        Do not add XML tags or any other special characters, just follow the output template.
        Use the examples as a guide and follow the instructions.
    """
    task_instruction = re.sub(r'\s{2,}', '\n', task_instruction).strip()
    task_prompt = f"<instruction>\n{task_instruction}\n</instruction>"

    output_instruction = f"""
        [LB_SOURCE]\n{{{src_lang_full} text with line-break(s)}}\n[/LB_SOURCE]
        [LB_TARGET]\n{{{tgt_lang_full} text with line-break(s)}}\n[/LB_TARGET]
    """
    output_instruction = re.sub(r'\s{2,}', '\n', output_instruction).strip()
    output_prompt = f"<output_template>\n{output_instruction}\n</output_template>"

    example_dict = {
        'ko': {'original': "안녕하세요. 제 이름은 GPT-4o입니다.", 'broken': "안녕하세요. 제 이름은\nGPT-4o입니다."},
        'en': {'original': "Hello. My name is GPT-4o.", 'broken': "Hello. My name is\nGPT-4o."},
        'ja': {'original': "こんにちは。私の名前はGPT-4oです。", 'broken': "こんにちは。私の名前は\nGPT-4oです。"},
        'zh': {'original': "你好。我的名字是GPT-4o。", 'broken': "你好。我的名字是\nGPT-4o。"},
    }

    example_instruction = f"""
        <example_input>
        <source><{src_lang_full}>\n{example_dict[src_lang_code]['original']}\n</{src_lang_full}></source>
        <target><{tgt_lang_full}>\n{example_dict[tgt_lang_code]['original']}\n</{tgt_lang_full}></target>
        </example_input>
        <example_output>
        [LB_SOURCE]\n{example_dict[src_lang_code]['broken']}\n[/LB_SOURCE]
        [LB_TARGET]\n{example_dict[tgt_lang_code]['broken']}\n[/LB_TARGET]
        </example_output>
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
        You have to generate two types of modified target text:
        (1) Maintained: Only the proper nouns should be maintained in {src_lang_full}. Other words should be translated into {tgt_lang_full}.
        (2) Translated: The proper nouns should be translated in {tgt_lang_full}. Other words should also be translated into {tgt_lang_full}.
        * Proper noun is the unique names of specific people, places, or things.
        * Their could be multiple proper nouns in the text.
        * If there is no proper noun in the text, the output should be "N/A" within the template.

        Do not change any other parts of the text, except for the proper nouns.
        Do not add XML tags or any other special characters, just follow the output template.
        Use the examples as a guide and follow the instructions.
    """
    task_instruction = re.sub(r'\s{2,}', '\n', task_instruction).strip()
    task_prompt = f"<instruction>\n{task_instruction}\n</instruction>"

    output_instruction = f"""
        [PN_MAINTAINED]\n{{{tgt_lang_full} target text with {src_lang_full} proper nouns, or N/A}}\n[/PN_MAINTAINED]
        [PN_TRANSLATED]\n{{{tgt_lang_full} target text with {tgt_lang_full} proper nouns, or N/A}}\n[/PN_TRANSLATED]
    """
    output_instruction = re.sub(r'\s{2,}', '\n', output_instruction).strip()
    output_prompt = f"<output_template>\n{output_instruction}\n</output_template>"

    example_dict = {
        'ko': {
            'source': "석가탑은 경주에 있는 대한민국의 대표적인 문화유산 중 하나입니다.",
            'target': {
                'en': "Seokgatap is one of the representative cultural heritages in 경주, South Korea.",
                'ja': "釈迦塔は경주にある韓国の代表的な文化遺産の一つです。",
                'zh': "释迦塔是韩国경주的代表性文化遗产之一。",
            },
            'maintained': {
                'en': "석가탑 is one of the representative cultural heritages in 경주, 대한민국.",
                'ja': "석가탑は경주にある대한민국の代表的な文化遺産の一つです。",
                'zh': "석가탑是대한민국경주的代表性文化遗产之一。",
            },
            'translated': {
                'en': "Seokgatap is one of the representative cultural heritages in Gyeongju, South Korea.",
                'ja': "釈迦塔は韓国の代表的な文化遺産の一つである慶州にあります。",
                'zh': "释迦塔是韩国慶州的代表性文化遗产之一。",
            }
        },
        'en': {
            'source': "The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor in New York City.",
            'target': {'ko': "자유의 여신상은 New York City 뉴욕 항구의 Liberty Island에 있는 거대한 신고전주의 조각입니다."},
            'maintained': {'ko': "The Statue of Liberty는 New York City New York Harbor의 Liberty Island에 있는 거대한 신고전주의 조각입니다."},
            'translated': {'ko': "자유의 여신상은 뉴욕시 뉴욕 항구의 자유의 섬에 있는 거대한 신고전주의 조각입니다."},
        },
        'ja': {
            'source': "東京タワーは東京都港区芝公園にある観光名所です。",
            'target': {'ko': "東京タワー는 도쿄도 미나토구 시바 공원에 있는 관광 명소입니다."},
            'maintained': {'ko': "東京タワー는 東京도 港구 芝公園에 있는 관광 명소입니다."},
            'translated': {'ko': "도쿄 타워는 도쿄도 미나토구 시바 공원에 있는 관광 명소입니다."},
        },
        'zh': {
            'source': "长城是中国的标志性建筑之一，位于北京市。",
            'target': {'ko': "长城은 중국의 상징적인 건물 중 하나로, 베이징에 위치해 있습니다."},
            'maintained': {'ko': "长城은 中国 北京에 위치한 상징적인 건물 중 하나입니다."},
            'translated': {'ko': "만리장성은 중국 베이징에 위치한 상징적인 건물 중 하나입니다."},
        },
    }

    example_instruction = f"""
        <example_input>
        <source><{src_lang_full}>\n{example_dict[src_lang_code]['source']}\n</{src_lang_full}></source>
        <target><{tgt_lang_full}>\n{example_dict[src_lang_code]['target'][tgt_lang_code]}\n</{tgt_lang_full}></target>
        </example_input>
        <example_output>
        [PN_MAINTAINED]\n{example_dict[src_lang_code]['maintained'][tgt_lang_code]}\n[/PN_MAINTAINED]
        [PN_TRANSLATED]\n{example_dict[src_lang_code]['translated'][tgt_lang_code]}\n[/PN_TRANSLATED]
        </example_output>
    """
    example_instruction = re.sub(r'\s{2,}', '\n', example_instruction).strip()
    example_prompt = f"<example>\n{example_instruction}\n</example>"

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
        * The structure and vocabulary of the target text can be significantly changed, based on the specified style.
        * However, unnecessary addition or deletion of information never be allowed.

        Do not add XML tags or any other special characters, just follow the output template.
        Use the examples as a guide and follow the instructions.
    """
    task_instruction = re.sub(r'\s{2,}', '\n', task_instruction).strip()
    task_prompt = f"<instruction>\n{task_instruction}\n</instruction>"

    output_instruction = f"""
        [ST_FORMAL]\n{{{tgt_lang_full} translation in formal style}}\n[/ST_FORMAL]
        [ST_INFORMAL]\n{{{tgt_lang_full} translation in informal style}}\n[/ST_INFORMAL]
    """
    output_instruction = re.sub(r'\s{2,}', '\n', output_instruction).strip()
    output_prompt = f"<output_template>\n{output_instruction}\n</output_template>"

    example_dict = {
        'ko': {
            'source': "정부는 이번 주말에 코로나 백신 접종을 받으러 오는 사람들에게 무료로 마스크를 제공할 예정이라고 밝혔다.",
            'target': {
                'formal': "당국은 금주 주말 코로나 백신 접종을 위해 방문하는 이들에게 무료 마스크를 제공할 예정이라 공지했다.",
                'informal': "정부에서 이번 주말에 코로나 백신 맞으러 오는 사람들한테 공짜로 마스크 나눠준다고 했어요.",
            }
        },
        'en': {
            'source': "The authorities have announced their plan to provide free masks to people who come for their COVID-19 vaccine this weekend.",
            'target': {
                'formal': "The authorities have announced their plan to distribute free masks to individuals who arrive for their COVID-19 vaccination this weekend.",
                'informal': "The government said they'll give free masks to people coming to get their COVID-19 vaccine this weekend.",
            }
        },
        'ja': {
            'source': "当局は今週末にCOVID-19ワクチンを接種しに来る人々に無料でマスクを提供する予定だと発表しました。",
            'target': {
                'formal': "当局は、今週末のCOVID-19ワクチン接種のために訪れる人々に無料でマスクを配布する予定だと発表しました。",
                'informal': "政府は今週末、ワクチン打ちに来る人にタダでマスクを配るって言ってたよ。",
            }
        },
        'zh': {
            'source': "当局宣布计划在本周末为前来接种COVID-19疫苗的人们免费提供口罩。",
            'target': {
                'formal': "当局宣布，他们计划在本周末为来接种COVID-19疫苗的民众免费发放口罩。",
                'informal': "政府说这个周末会给来打疫苗的人发免费口罩。",
            }
        }
    }

    example_instruction = f"""
        <example_input>
        <source><{src_lang_full}>\n{example_dict[src_lang_code]['source']}\n</{src_lang_full}></source>
        <target><{tgt_lang_full}>\n{example_dict[tgt_lang_code]['source']}\n</{tgt_lang_full}></target>
        </example_input>
        <example_output>
        [ST_FORMAL]\n{example_dict[tgt_lang_code]['target']['formal']}\n[/ST_FORMAL]
        [ST_INFORMAL]\n{example_dict[tgt_lang_code]['target']['informal']}\n[/ST_INFORMAL]
        </example_output>
    """
    example_instruction = re.sub(r'\s{2,}', '\n', example_instruction).strip()
    example_prompt = f"<example>\n{example_instruction}\n</example>"

    source_prompt = f"<source><{src_lang_full}>\n{src_text}\n</{src_lang_full}></source>"
    target_prompt = f"<target><{tgt_lang_full}>\n{tgt_text}\n</{tgt_lang_full}></target>"

    user_prompt = f"{task_prompt}\n\n{output_prompt}\n\n{example_prompt}\n\n{source_prompt}\n{target_prompt}"

    return user_prompt


def generate_single(src_text, tgt_text, prompt_type, src_lang_code='en', tgt_lang_code='ko', gpt_version='gpt-4o-mini'):
    if prompt_type == 'linebreak':
        user_prompt = get_user_prompt_linebreak(src_text, tgt_text, src_lang_code, tgt_lang_code)
    elif prompt_type == 'propernoun':
        user_prompt = get_user_prompt_propernoun_contrastive(src_text, tgt_text, src_lang_code, tgt_lang_code)
    elif prompt_type == 'style':
        user_prompt = get_user_prompt_style_contrastive(src_text, tgt_text, src_lang_code, tgt_lang_code)
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")

    gpt_generator = GptGenerator(api_key=OPENAI_CLIENT_KEY_TMAXNLP)
    output = gpt_generator.generate(user_prompt, gpt_version=gpt_version, src_lang_code=src_lang_code, tgt_lang_code=tgt_lang_code)

    print(f"<<<SYSTEM PROMPT>>>\n{gpt_generator.get_gpt_system_prompt(src_lang_code, tgt_lang_code)}")
    print(f"\n<<<USER PROMPT>>>\n{user_prompt}")
    print(f"\n<<<GENERATED OUTPUT>>>\n{output}")


def make_jsonl_request(df, prompt_type, jsonl_path, gpt_version='gpt-4o-mini'):
    user_prompt_func = {
        'linebreak': get_user_prompt_linebreak,
        'propernoun': get_user_prompt_propernoun_contrastive,
        'style': get_user_prompt_style_contrastive,
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

        user_prompt = user_prompt_func[prompt_type](src_text, tgt_text, src_lang, tgt_lang)
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


def upload_jsonl_request(client, df, prompt_type, jsonl_path, gpt_version): 
    if not os.path.exists(jsonl_path):
        make_jsonl_request(df, prompt_type, jsonl_path, gpt_version)
    else:
        raise ValueError(f"JSONL file already exists: {jsonl_path}")

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

    en_path = '../../inference/results/prime-cleansing/sft/en_not_prime_dpo_sample.csv'
    ja_path = '../../inference/results/prime-cleansing/sft/ja_not_prime_dpo_sample.csv'
    zh_path = '../../inference/results/prime-cleansing/sft/zh_not_prime_dpo_sample.csv'
    en_df = pd.read_csv(en_path)
    ja_df = pd.read_csv(ja_path)
    zh_df = pd.read_csv(zh_path)
    en_df = en_df.sample(frac=1, random_state=42).reset_index(drop=True)
    ja_df = ja_df.sample(frac=1, random_state=42).reset_index(drop=True)
    zh_df = zh_df.sample(frac=1, random_state=42).reset_index(drop=True)
    df_dict = {'en': en_df, 'ja': ja_df, 'zh': zh_df}

    gen_dict = {
        'normal': {
            'en': {'size': 275, 'jsonl_path': './gpt_dpo_normal.jsonl'},
            'ja': {'size': 220, 'jsonl_path': './gpt_dpo_normal.jsonl'},
            'zh': {'size': 220, 'jsonl_path': './gpt_dpo_normal.jsonl'},
        },
        'linebreak': {
            'en': {'size': 625, 'jsonl_path': './gpt_dpo_linebreak_request.jsonl'},
            'ja': {'size': 500, 'jsonl_path': './gpt_dpo_linebreak_request.jsonl'},
            'zh': {'size': 500, 'jsonl_path': './gpt_dpo_linebreak_request.jsonl'},
        },
        'propernoun': {
            'en': {'size': 1875, 'jsonl_path': './gpt_dpo_propernoun_request.jsonl'},
            'ja': {'size': 1500, 'jsonl_path': './gpt_dpo_propernoun_request.jsonl'},
            'zh': {'size': 1500, 'jsonl_path': './gpt_dpo_propernoun_request.jsonl'},
        },
        'style': {
            'en': {'size': 1250, 'jsonl_path': './gpt_dpo_style_request.jsonl'},
            'ja': {'size': 1000, 'jsonl_path': './gpt_dpo_style_request.jsonl'},
            'zh': {'size': 1000, 'jsonl_path': './gpt_dpo_style_request.jsonl'},
        },
    }

    file_infos = {}
    start_infos = {'en': 0, 'ja': 0, 'zh': 0}
    for prompt_type in ['linebreak', 'propernoun', 'style', 'normal']:
        params = gen_dict[prompt_type]
        df_batch = pd.DataFrame()

        for lang, df in df_dict.items():
            df_size = params[lang]['size']
            jsonl_path = params[lang]['jsonl_path']

            for source in df['data_source'].unique():
                df_source = df[df['data_source'] == source].iloc[start_infos[lang]:start_infos[lang]+df_size]
                df_batch_src2tgt = df_source.iloc[:len(df_source)//2]
                df_batch_tgt2src = reverse_direction(df_source.iloc[len(df_source)//2:])
                df_batch_subset = pd.concat([df_batch_src2tgt, df_batch_tgt2src], ignore_index=True)
                df_batch = pd.concat([df_batch, df_batch_subset], ignore_index=True)

            start_infos[lang] += df_size
        
        if prompt_type == 'normal':
            df_batch.to_csv('./gpt_dpo_normal.csv', index=False)
            continue
        # gpt_version = 'gpt-4o' if prompt_type == 'propernoun' else gpt_version
        file_info = upload_jsonl_request(client, df_batch, prompt_type, jsonl_path, gpt_version)
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
    # # Single generation test
    # src_lang = 'en'
    # tgt_lang = 'ko'
    # src_text = "The Eiffel Tower is an iron tower located on the Champ de Mars in Paris, France."
    # tgt_text = "에펠탑은 파리의 샹데마르 광장에 위치한 철제 탑이다."
    # prompt_type = 'style'
    # gpt_version = 'gpt-4o'
    # output = generate_single(src_text, tgt_text, prompt_type, src_lang, tgt_lang, gpt_version=gpt_version)

    # Batch inference
    make_upload_inference()
    sleep(10)
    check_batches(3)

    # # Download Batch
    # download_responses('file-sgMBPNKUmSYrw5dAbhg3KCse', './gpt_dpo_style_response_v2.jsonl')

    # client = OpenAI(api_key=OPENAI_CLIENT_KEY_TMAXNLP)

    # # Cancel batch
    # batch_num = 0
    # batch_id = list(client.batches.list())[batch_num].id
    # client.batches.cancel(batch_id)
