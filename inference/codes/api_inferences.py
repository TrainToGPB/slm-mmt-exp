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
import requests
from tqdm import tqdm

# third-party
from googletrans import Translator as GoogleTranslator
from deepl import Translator as DeeplTranslator
import pandas as pd

# custom
sys.path.append('./')
from api_secret import PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET
from api_secret import (
    DEEPL_CLIENT_KEY_0,
    DEEPL_CLIENT_KEY_1,
)


DEEPL_LANGCODES = {
    'en': 'EN-US',
    'ko': 'KO',
}


class PapagoTranslator:
    def __init__(self, client_id, client_secret):
        """
        Initializes a new instance of the PapagoTranslator class.

        Args:
            client_id (str): The client ID for the Papago API.
            client_secret (str): The client secret for the Papago API.
        """
        self.url = 'https://openapi.naver.com/v1/papago/n2mt'
        self.headers = headers = {
            'Content-Type': 'application/json',
            'X-Naver-Client-Id': client_id,
            'X-Naver-Client-Secret': client_secret
        }

    def translate(self, src_lang, tgt_lang, text):
        """
        Translates the given text from the source language to the target language.

        Args:
            src_lang (str): The source language code.
            tgt_lang (str): The target language code.
            text (str): The text to be translated.

        Returns:
            str: The translated text.
        """
        self.data = {
            'source': src_lang,
            'target': tgt_lang,
            'text': text
        }
        response = requests.post(self.url, json.dumps(self.data), headers=self.headers)
        translation = eval(response.text)['message']['result']['translatedText']
        return translation


def papato_translate_text(text, src_lang='en', tgt_lang='ko'):
    """
    Translates a single text from English to Korean using Papago Translator.

    Args:
    - text (str): The text to be translated.

    Returns:
    - papago_translation (str): The translated text.
    """
    papago_translator = PapagoTranslator(PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET)
    papago_translation = papago_translator.translate(src_lang=src_lang, tgt_lang=tgt_lang, text=text)
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
    deepl_translator = DeeplTranslator(DEEPL_CLIENT_KEY)
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


def papago_translate_df(df, client_id=PAPAGO_CLIENT_ID, client_secret=PAPAGO_CLIENT_SECRET, print_result=True):
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
        if len(translations) == len(df['ko'].tolist()):
            print("All data are translated already.")
            return df
        start_idx = df['papago_trans'].isnull().idxmax()
    else:
        translations = []
        start_idx = 0

    tqdm_iterator = tqdm(df.iloc[start_idx:].iterrows(), total=len(df) - start_idx, desc='DeepL translating')
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
            print("Error Occured: Papago")
            translations.extend([None] * (len(df[src_col]) - len(translations)))
            break

        try:
            translation = translator.translate(src_lang=src_lang, tgt_lang=tgt_lang, text=text)
        except:
            error_occured = True
            translation = None

        if print_result:
            print(f"[INPUT] {text}")
            print(f"[OUTPUT] {translation}")
        translations.append(translation)

    df['papago_trans'] = translations

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
        start_idx = df['google_trans'].isnull().idxmax()
    else:
        translations = []
        start_idx = 0

    tqdm_iterator = tqdm(df.iloc[start_idx:].iterrows(), total=len(df) - start_idx, desc='DeepL translating')
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
            print("Error Occured: Google")
            translations.extend([None] * (len(df[src_col]) - len(translations)))
            break

        try:
            translation = translator.translate(src=src_lang, dest=tgt_lang, text=text).text
        except:
            error_occurred = True
            translation = None

        if print_result:
            print(f"[INPUT] {text}")
            print(f"[OUTPUT] {translation}")
        translations.append(translation)

    df['google_trans'] = translations

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
        start_idx = df['deepl_trans'].isnull().idxmax()
    else:
        translations = []
        start_idx = 0

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
            print("Error Occured: DeepL")
            translations.extend([None] * (len(df[src_col]) - len(translations)))
            break

        try:
            translation = translator.translate_text(target_lang=DEEPL_LANGCODES[tgt_lang], text=text)
        except:
            error_occurred = True
            translation = None

        if print_result:
            print(f"[INPUT] {text}")
            print(f"[OUTPUT] {translation}")
        translations.append(translation)

    df['deepl_trans'] = translations

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
        'sparta': '../results/train_sparta_bidir_inferenced.csv'
    }
    file_path = file_path_dict[args.dataset]

    # deepl
    client_keys = [
        DEEPL_CLIENT_KEY_0, # 세형
        DEEPL_CLIENT_KEY_1, # 성환님
    ]
    eval_df = pd.read_csv(file_path)
    for client_key in client_keys:
        eval_df = translate_df(eval_df, translator='deepl', client_key=client_key, print_result=True)
        eval_df.to_csv(file_path, index=False)

    # eval_df = pd.read_csv(file_path)
    # eval_df = translate_df(eval_df, translator=args.translator) # Add client information if necessary
    # eval_df.to_csv(file_path, index=False) # Modify save path if necessary


if __name__ == '__main__':
    main()
