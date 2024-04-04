"""
This module contains functions for translating texts using various translation APIs.

The following classes are available:
- PapagoTranslator: A class for translating texts using the Papago Translator API.

The following functions are available:
- papato_translate_text: Translates a single text from English to Korean using Papago Translator.
- google_translate_text: Translates a single text from English to Korean using Google Translate API.
- deepl_translate_text: Translates a single text using the DeepL translation API.
- translate_single_text: Translates a single text using the specified translator.
- papago_translate_df: Translates the 'en' column of a DataFrame using the Papago Translator API.
- google_translate_df: Translates the 'en' column of a DataFrame from English to Korean using Google Translate API.
- deepl_translate_df: Translates the 'en' column of a DataFrame using the DeepL API.
- translate_df: Translates a DataFrame using the specified translator.

Examples:
    & python api_inferences.py --dataset aihub --translator papago
    & python api_inferences.py --dataset flores --translator deepl

Notes:
    - The 'api_secret.py' file should be located in the same directory as this module.
    - The 'api_secret.py' file should contain the following variables:
        - PAPAGO_CLIENT_ID
        - PAPAGO_CLIENT_SECRET
        - DEEPL_CLIENT_KEY
"""
# built-in
import sys
import json
from urllib import parse, request
from tqdm import tqdm
from datetime import datetime

# third-party
from googletrans import Translator as GoogleTranslator
from deepl import Translator as DeeplTranslator
import pandas as pd

# custom
sys.path.append('./')
from api_secret import (
    PAPAGO_CLIENT_ID_0, 
    PAPAGO_CLIENT_SECRET_0,
)
from api_secret import (
    DEEPL_CLIENT_KEY_0,
    DEEPL_CLIENT_KEY_1,
    DEEPL_CLIENT_KEY_2,
)


DEEPL_LANGCODES = {
    'en': 'EN-US',
    'ko': 'KO',
}


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


def papato_translate_text(text, src_lang='en', tgt_lang='ko'):
    """
    Translates a single text from English to Korean using Papago Translator.

    Args:
    - text (str): The text to be translated.

    Returns:
    - papago_translation (str): The translated text.
    """
    papago_translator = PapagoTranslator(PAPAGO_CLIENT_ID_0, PAPAGO_CLIENT_SECRET_0)
    papago_translation = papago_translator.translate(text=text, src_lang=src_lang, tgt_lang=tgt_lang)
    return papago_translation


def google_translate_text(text, src_lang='en', tgt_lang='ko'):
    """
    Translates a single text from English to Korean using Google Translate API.

    Args:
    - text (str): The text to be translated.

    Returns:
    - google_translation (str): The translated text.
    """
    google_translator = GoogleTranslator()
    google_translation = google_translator.translate(src=src_lang, dest=tgt_lang, text=text).text
    return google_translation


def deepl_translate_text(text, tgt_lang='ko'):
    """
    Translates a single text using the DeepL translation API.

    Args:
    - text (str): The text to be translated.

    Returns:
    - deepl_translation (str): The translated text.
    """
    deepl_translator = DeeplTranslator(DEEPL_CLIENT_KEY_0)
    deepl_translation = deepl_translator.translate_text(target_lang=DEEPL_LANGCODES[tgt_lang], text=text)
    return deepl_translation


def translate_single_text(text, translator='google'):
    """
    Translates a single text using the specified translator.

    Args:
    - text (str): The text to be translated.
    - translator (str, optional): The translator to be used. Defaults to 'google'.

    Returns:
    - translation (str): The translated text.
    """
    trans_dict = {
        'papago': papato_translate_text,
        'google': google_translate_text,
        'deepl': deepl_translate_text
    }
    translatior = trans_dict[translator]
    translation = translatior(text)
    return translation


def papago_translate_df(df, client_id=PAPAGO_CLIENT_ID_0, client_secret=PAPAGO_CLIENT_SECRET_0, print_result=True):
    """
    Translates the 'en' column of a DataFrame using the Papago Translator API.

    Args:
    - df (pandas.DataFrame): The DataFrame to be translated.
    - client_id (str): The client ID for the Papago Translator API. Default is PAPAGO_CLIENT_ID_0.
    - client_secret (str): The client secret for the Papago Translator API. Default is PAPAGO_CLIENT_SECRET_0.

    Returns:
    - translated_df (pandas.DataFrame): The DataFrame with the translated
    """
    
    translator = PapagoTranslator(client_id, client_secret)
    error_occured = False

    if 'papago_trans' in df.columns:
        translations = df['papago_trans'].dropna().tolist()
        if len(translations) == len(df):
            print("All data are translated already with Papago.")
            return df
        start_idx = df['papago_trans'].isnull().idxmax()
    else:
        translations = []
        start_idx = 0

    if 'papago_time' in df.columns:
        elapsed_times = df['papago_time'].dropna().tolist()
    else:
        elapsed_times = []

    tqdm_iterator = tqdm(df.iloc[start_idx:].iterrows(), total=len(df) - start_idx, desc='Papago translating')
    for _, row in tqdm_iterator:
        if 'src' in df.columns:
            text = row['src']
        else:
            text = row['en']
        if 'direction' in df.columns:
            src_lang, tgt_lang = row['direction'].split('-')
            src_col = 'src'
        else:
            src_lang, tgt_lang = 'en', 'ko'
            src_col = 'en'

        if error_occured:
            print(f"Error Occured - Papago:\n{error_message}")
            translations.extend([None] * (len(df[src_col]) - len(translations)))
            elapsed_times.extend([None] * (len(df[src_col]) - len(elapsed_times)))
            break

        try:
            start_time = datetime.now()
            translation = translator.translate(text=text, src_lang=src_lang, tgt_lang=tgt_lang)
            end_time = datetime.now()
            elapsed_time = round((end_time - start_time).total_seconds() * 1000, 1)
            if print_result:
                print(f"[INPUT] {text}")
                print(f"[OUTPUT] {translation}")
                print(f"[ELAPSED TIME] {elapsed_time} ms")

        except Exception as e:
            error_message = e
            error_occured = True
            translation = None
            elapsed_time = None

        translations.append(translation)
        elapsed_times.append(elapsed_time)

    df['papago_trans'] = translations
    df['papago_time'] = elapsed_times

    return df


def google_translate_df(df, print_result=True):
    """
    Translates the 'en' column of a DataFrame from English to Korean using Google Translate API.

    Args:
    - df (pandas.DataFrame): The DataFrame containing the 'en' column to be translated.

    Returns:
    - df (pandas.DataFrame): The DataFrame with an additional '
    """
    translator = GoogleTranslator()
    error_occurred = False

    if 'google_trans' in df.columns:
        translations = df['google_trans'].dropna().tolist()
        if len(translations) == len(df):
            print("All data are translated already with Google.")
            return df
        start_idx = df['google_trans'].isnull().idxmax()
    else:
        translations = []
        start_idx = 0

    if 'google_time' in df.columns:
        elapsed_times = df['google_time'].dropna().tolist()
    else:
        elapsed_times = []

    tqdm_iterator = tqdm(df.iloc[start_idx:].iterrows(), total=len(df) - start_idx, desc='Google translating')
    for _, row in tqdm_iterator:
        if 'src' in df.columns:
            text = row['src']
        else:
            text = row['en']
        if 'direction' in df.columns:
            src_lang, tgt_lang = row['direction'].split('-')
            src_col = 'src'
        else:
            src_lang, tgt_lang = 'en', 'ko'
            src_col = 'en'

        if error_occurred:
            print(f"Error Occured - Google:\n{error_message}")
            translations.extend([None] * (len(df[src_col]) - len(translations)))
            elapsed_times.extend([None] * (len(df[src_col]) - len(elapsed_times)))
            break

        try:
            start_time = datetime.now()
            translation = translator.translate(src=src_lang, dest=tgt_lang, text=text).text
            end_time = datetime.now()
            elapsed_time = round((end_time - start_time).total_seconds() * 1000, 1)
            if print_result:
                print(f"[INPUT] {text}")
                print(f"[OUTPUT] {translation}")
                print(f"[ELAPSED TIME] {elapsed_time} ms")

        except Exception as e:
            error_message = e
            error_occurred = True
            translation = None
            elapsed_time = None
        
        translations.append(translation)
        elapsed_times.append(elapsed_time)

    df['google_trans'] = translations
    df['google_time'] = elapsed_times

    return df


def deepl_translate_df(df, client_key=DEEPL_CLIENT_KEY_0, print_result=True):
    """
    Translates the source column of a DataFrame using the DeepL API.

    Args:
    - df (pandas.DataFrame): The DataFrame containing the 'en' column to be translated.
    - client_key (str): The client key for accessing the DeepL API. Defaults to DEEPL_CLIENT_KEY_0.

    Returns:
    - df (pandas.DataFrame): The DataFrame with an additional column containing the translated texts.
    """

    translator = DeeplTranslator(client_key)
    error_occurred = False

    if 'deepl_trans' in df.columns:
        translations = df['deepl_trans'].dropna().tolist()
        if len(translations) == len(df):
            print("All data are translated already with DeepL.")
            return df
        start_idx = df['deepl_trans'].isnull().idxmax()
    else:
        translations = []
        start_idx = 0

    if 'deepl_time' in df.columns:
        elapsed_times = df['deepl_time'].dropna().tolist()
    else:
        elapsed_times = []

    tqdm_iterator = tqdm(df.iloc[start_idx:].iterrows(), total=len(df) - start_idx, desc='DeepL translating')
    for _, row in tqdm_iterator:
        if 'src' in df.columns:
            text = row['src']
        else:
            text = row['en']
        if 'direction' in df.columns:
            _, tgt_lang = row['direction'].split('-')
            src_col = 'src'
        else:
            _, tgt_lang = 'en', 'ko'
            src_col = 'en'

        if error_occurred:
            print(f"Error Occured - DeepL:\n{error_message}")
            translations.extend([None] * (len(df[src_col]) - len(translations)))
            elapsed_times.extend([None] * (len(df[src_col]) - len(elapsed_times)))
            break

        try:
            start_time = datetime.now()
            translation = translator.translate_text(target_lang=DEEPL_LANGCODES[tgt_lang], text=text)
            end_time = datetime.now()
            elapsed_time = round((end_time - start_time).total_seconds() * 1000, 1)
            if print_result:
                print(f"[INPUT] {text}")
                print(f"[OUTPUT] {translation}")
                print(f"[ELAPSED TIME] {elapsed_time} ms")

        except Exception as e:
            error_message = e
            error_occurred = True
            translation = None
            elapsed_time = None

        translations.append(translation)
        elapsed_times.append(elapsed_time)

    df['deepl_trans'] = translations
    df['deepl_time'] = elapsed_times

    return df


def translate_df(df, translator='google', **kwargs):
    """
    Translates a DataFrame using the specified translator.

    Parameters:
    - df: pandas DataFrame
        The DataFrame to be translated.
    - translator: str, optional (default: 'google')
        The translator to be used. Available options: 'papago', 'google', 'deepl'.
    - **kwargs: keyword arguments, including client information if necessary.
        Additional arguments to be passed to the translator function.
        For Papago, 'client_id' and 'client_secret' are required.
        For DeepL, 'client_key' is required.

    Returns:
    - translated_df (pandas.DataFrame): The DataFrame with the translated texts.

    Examples:
    >>> df = pd.DataFrame({'en': ['Hello', 'World']})
    >>> translated_df = translate_df(df, translator='papago', client_id='YOUR_CLIENT_ID', client_secret='YOUR_CLIENT_SECRET')
    """
    trans_dict = {
        'papago': papago_translate_df,
        'google': google_translate_df,
        'deepl': deepl_translate_df
    }
    translator = trans_dict[translator]
    translated_df = translator(df, **kwargs)
    return translated_df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='aihub', help='Dataset to inference')
    parser.add_argument('--translator', type=str, default='google', help='Translator to use: papago, google, deepl')
    args = parser.parse_args()

    file_path_dict = {
        'aihub': '../results/test_tiny_uniform100_inferenced.csv',
        'flores': '../results/test_flores_inferenced.csv',
        'sparta': '../results/test_sparta_bidir_inferenced.csv',
        'sample': '../results/sample.csv',
    }
    file_path = file_path_dict[args.dataset]

    # # google
    # eval_df = pd.read_csv(file_path)
    # eval_df = translate_df(eval_df, translator='google', print_result=True)
    # eval_df.to_csv(file_path, index=False)

    # papago
    papago_client_ids = [
        PAPAGO_CLIENT_ID_0, # 세형
    ]
    papago_client_secrets = [
        PAPAGO_CLIENT_SECRET_0, # 세형
    ]
    eval_df = pd.read_csv(file_path)
    for client_id, client_secret in zip(papago_client_ids, papago_client_secrets):
        eval_df = translate_df(eval_df, translator='papago', client_id=client_id, client_secret=client_secret, print_result=True)
        eval_df.to_csv(file_path, index=False)

    # # deepl
    # deepl_client_keys = [
    #     # DEEPL_CLIENT_KEY_0, # 세형
    #     # DEEPL_CLIENT_KEY_1, # 성환님
    #     DEEPL_CLIENT_KEY_2, # 현경님
    # ]
    # eval_df = pd.read_csv(file_path)
    # for client_key in deepl_client_keys:
    #     eval_df = translate_df(eval_df, translator='deepl', client_key=client_key, print_result=True)
    #     eval_df.to_csv(file_path, index=False)


if __name__ == '__main__':
    main()
