import os
import json
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd


def flatten_json(y):
    out = {}
    separator = '--'
    def flatten(x, name=''):
        if isinstance(x, dict):
            for key in x:
                flatten(x[key], f"{name}{key}{separator}")
        elif isinstance(x, list):
            for item in x:
                flatten(item, name)
        else:
            out[name[:-len(separator)]] = x  # 마지막 구분자 제거

    flatten(y)

    return out


def make_file_path(data_num, split, data_paths, root_dir):
    return f'{root_dir}/{data_paths[data_num]}/{split}'


def process_file(file_info):
    file_path, file = file_info
    if file.endswith('.json'):
        try:
            with open(f'{file_path}/{file}', 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
            if "data" in data and isinstance(data["data"], list):
                flattened_data = [flatten_json(item) for item in data["data"]]
            elif "paragraph" in data:
                flattened_data = []
                for paragraph in data["paragraph"]:
                    info = flatten_json(paragraph["info"])
                    for sentence in paragraph["sentences"]:
                        sentence_flatten = flatten_json(sentence)
                        merged_data = {**info, **sentence_flatten}
                        flattened_data.append(merged_data)
            else:
                flattened_data = [flatten_json(data)]
            return pd.DataFrame(flattened_data)
        
        except Exception as e:
            print(f'Error: {file_path}/{file} - {e}')
            return pd.DataFrame()

    elif file.endswith('.csv'):
        return pd.read_csv(f'{file_path}/{file}')

    return pd.DataFrame()


def make_df(data_num, split, data_paths, root_dir):
    file_path = make_file_path(data_num, split, data_paths, root_dir)
    files = os.listdir(file_path)
    
    with Pool() as pool:
        df_list = list(tqdm(pool.imap(process_file, [(file_path, file) for file in files]), total=len(files), desc=f'{data_num} {split.upper()}'))

    df = pd.concat(df_list, ignore_index=True)
    return df


def integrate_data(data_paths, root_dir, save_root_dir):
    for data_num in tqdm(data_paths.keys(), desc='Data integrating'):
        if data_num != 71591:
            continue
        train_df = make_df(data_num, 'train', data_paths, root_dir)
        train_df.to_csv(f'{save_root_dir}/train_{data_num}.csv', index=False)
        train_df_sample = train_df.iloc[:10]
        train_df_sample.to_csv(f'{save_root_dir}/train_{data_num}_sample.csv', index=False)

        eval_df = make_df(data_num, 'validation', data_paths, root_dir)
        eval_df.to_csv(f'{save_root_dir}/validation_{data_num}.csv', index=False)


if __name__ == '__main__':
    root_dir = '/home/ubuntu/sehyeong/tmax-enko-mt/translation_datasets/mmt/aihub_integration/raw'
    data_paths = {
        127: '127. 일반_일어',
        128: '128. 사회과학_중어',
        129: '129. 기술과학2_중어',
        546: '546. 일상생활및구어체',
        71262: '71262. 식품_중어',
        71263: '71263. 방송콘텐츠2',
        71411: '71411. 발화유형별',
        # 71428: '71428. 관광지',
        71493: '71493. 기술과학1',
        71496: '71496. 기초과학',
        71498: '71498. 인문학',
        71524: '71524. 낭독체',
        71591: '71591. 방송콘텐츠1',
        71593: '71593. 종합'
    }
    save_root_dir = '/home/ubuntu/sehyeong/tmax-enko-mt/translation_datasets/mmt/aihub_integration/integrated_subsets'

    integrate_data(data_paths, root_dir, save_root_dir)
