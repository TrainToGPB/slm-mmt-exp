import re
from tqdm import tqdm
from collections import defaultdict

import pandas as pd


def extract_response(df):
    data = defaultdict(list)
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Extracting responses'):
        id = row['custom_id']
        lang_pair = id.split('-')[1]
        direction = lang_pair[:2] + '-' + lang_pair[2:]

        response = row['response']['body']['choices'][0]['message']['content']
        response = postprocess_response(response)

        data['id'].append(id)
        data['direction'].append(direction)
        data['response'].append(response)

    responses = pd.DataFrame(data)

    return responses


def postprocess_response(response):
    if response.startswith('<output_template>'):
        response = response.replace('<output_template>', '')
    if response.endswith('</output_template>'):
        response = response.replace('</output_template>', '')
    response = response.strip()
    
    return response


def cleanse_response(response, data_type):
    def extract_tagged_text_for_responses(text, tag):
        pattern = fr'\[{tag}\](.*?)\[/{tag}\]'
        match = re.search(pattern, text, re.DOTALL)
        extracted_text = match.group(1).strip() if match else None
        return extracted_text
    
    if data_type == 'linebreak':
        text_source = extract_tagged_text_for_responses(response, 'LB_SOURCE')
        text_target = extract_tagged_text_for_responses(response, 'LB_TARGET')
        if text_source is None or text_target is None:
            return None
        response_pair = (text_source, text_target)
        return response_pair

    elif data_type == 'propernoun':
        if 'n/a' in response.lower():
            return None
        text_maintained = extract_tagged_text_for_responses(response, 'PN_MAINTAINED')
        text_translated = extract_tagged_text_for_responses(response, 'PN_TRANSLATED')
        if text_maintained is None or text_translated is None:
            return None
        if text_maintained == text_translated:
            return None
        response_pair = (text_maintained, text_translated)
        return response_pair
    
    elif data_type == 'style':
        text_formal = extract_tagged_text_for_responses(response, 'ST_FORMAL')
        text_informal = extract_tagged_text_for_responses(response, 'ST_INFORMAL')
        if text_formal is None or text_informal is None:
            return None
        response_pair = (text_formal, text_informal)
        return response_pair

def main():
    data_paths = [
        # 'linebreak', 
        # 'propernoun', 
        'style'
    ]

    # # 1. Extract responses from JSONL files
    # for data_path in data_paths:
    #     full_path = f'./gpt_dpo_{data_path}_response_v2.jsonl'
    #     df = pd.read_json(full_path, lines=True)
    #     responses = extract_response(df)
    #     responses.to_csv(f'./gpt_dpo_{data_path}_response_v2.csv', index=False)

    # 2. Cleanse responses
    for data_path in data_paths:
        if data_path == 'linebreak':
            text1 = 'source'
            text2 = 'target'
            continue
        elif data_path == 'propernoun':
            text1 = 'maintained'
            text2 = 'translated'
            continue
        elif data_path == 'style':
            text1 = 'formal'
            text2 = 'informal'

        responses = pd.read_csv(f'./gpt_dpo_{data_path}_response_v2.csv')
        for idx_res, row_res in responses.iterrows():
            response_pair = cleanse_response(row_res['response'], data_path)
            if response_pair is not None:
                responses.loc[idx_res, text1] = response_pair[0]
                responses.loc[idx_res, text2] = response_pair[1]
            else:
                responses.loc[idx_res, text1] = None
                responses.loc[idx_res, text2] = None

        responses = responses.drop(columns=['response'])
        responses = responses.dropna(subset=[text1, text2])
        responses.to_csv(f'./gpt_dpo_{data_path}_response_cleansed_v2.csv', index=False)


if __name__ == '__main__':
    main()
