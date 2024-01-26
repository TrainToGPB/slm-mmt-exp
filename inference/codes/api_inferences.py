import sys
import json
import requests
from tqdm import tqdm

from googletrans import Translator as GoogleTranslator
from deepl import Translator as DeeplTranslator
import pandas as pd

sys.path.append('./')
from api_secret import PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET
from api_secret import DEEPL_CLIENT_KEY


class PapagoTranslator:
    """
    A class for translating text using the Papago translation service.
    To use this class, you should make './api_secret.py' including your client id and secret.

    Parameters:
    - client_id (str): Naver Papago API client ID.
    - client_secret (str): Naver Papago API client secret.

    Methods:
    - translate(src_lang, tgt_lang, text): Translates the input text from the source language to the target language.
    """

    def __init__(self, client_id, client_secret):
        self.url = 'https://openapi.naver.com/v1/papago/n2mt'
        self.headers = headers = {
            'Content-Type': 'application/json',
            'X-Naver-Client-Id': client_id,
            'X-Naver-Client-Secret': client_secret
        }

    def translate(self, src_lang, tgt_lang, text):
        """
        Translates the input text from the source language to the target language using Papago translation API.
        Papago API's translation limit is 10000 'letters (not tokens)' per a day, and if you exceed the limit 'KeyError' will be asserted.
        To avoid breaking the code, this method returns 'None' when the 'KeyError' is asserted.

        Parameters:
        - src_lang (str): Source language code.
        - tgt_lang (str): Target language code.
        - text (str): Text to be translated.

        Returns:
        - translation (str): Translated text.
        """
        self.data = {
            'source': src_lang,
            'target': tgt_lang,
            'text': text
        }
        response = requests.post(self.url, json.dumps(self.data), headers=self.headers)
        translation = eval(response.text)['message']['result']['translatedText']
        return translation


def translate_sentence(text, api_type='google'):
    """
    Translates a single text using multiple translation services.

    Parameters:
    - text (str): Text to be translated.
    - api_type (str): Translation API to use.

    Returns:
    - translation (str): Translated text using API.
    """
    assert api_type in ['google', 'papago', 'deepl'], 'Wrong API type'
    
    if api_type == 'papago':
        translator = PapagoTranslator(PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET)
        translation = translator.translate(src_lang='en', tgt_lang='ko', text=text)

    elif api_type == 'google':
        translator = GoogleTranslator()
        translation = translator.translate(src='en', dest='ko', text=text).text

    elif api_type == 'deepl':
        translator = DeeplTranslator(DEEPL_CLIENT_KEY)
        translation = translator.translate_text(target_lang='KO', text=text)

    return translation


def papago_translate(df, client_id=PAPAGO_CLIENT_ID, client_secret=PAPAGO_CLIENT_SECRET):
    """
    Translate English text to Korean using the Papago translation service and update the DataFrame.

    Parameters:
    - df (DataFrame): DataFrame containing the 'en' column with English text to be translated.

    Returns:
    - df (DataFrame): Updated DataFrame with the 'papago_trans' column containing Korean translations.
    """
    translator = PapagoTranslator(client_id, client_secret)
    error_occured = False

    if 'papago_trans' in df.columns:
        translations = df['papago_trans'].dropna().tolist()
        start_idx = df['papago_trans'].isnull().idxmax()
    else:
        translations = []
        start_idx = 0

    for text in tqdm(df['en'][start_idx:]):
        if error_occured:
            print("Error Occured: Papago")
            translations.extend([None] * (len(df['en']) - len(translations)))
            break

        try:
            translation = translator.translate(src_lang='en', tgt_lang='ko', text=text)
        except:
            error_occured = True
            translation = None

        translations.append(translation)

    df['papago_trans'] = translations

    return df


def google_translate(df):
    """
    Translate English text to Korean using the Google Translate service and update the DataFrame.

    Parameters:
    - df (DataFrame): DataFrame containing the 'en' column with English text to be translated.

    Returns:
    - df (DataFrame): Updated DataFrame with the 'google_trans' column containing Korean translations.
    """
    translator = GoogleTranslator()
    error_occurred = False

    if 'google_trans' in df.columns:
        translations = df['google_trans'].dropna().tolist()
        start_idx = df['google_trans'].isnull().idxmax()
    else:
        translations = []
        start_idx = 0

    for text in tqdm(df['en'][start_idx:]):
        if error_occurred:
            print("Error Occured: Google")
            translations.extend([None] * (len(df['en']) - len(translations)))
            break

        try:
            translation = translator.translate(src='en', dest='ko', text=text).text
        except:
            error_occurred = True
            translation = None

        translations.append(translation)

    df['google_trans'] = translations

    return df


def deepl_translate(df, client_key=DEEPL_CLIENT_KEY):
    """
    Translate English text to Korean using the DeepL translation service and update the DataFrame.

    Parameters:
    - df (DataFrame): DataFrame containing the 'en' column with English text to be translated.

    Returns:
    - df (DataFrame): Updated DataFrame with the 'deepl_trans' column containing Korean translations.
    """
    translator = DeeplTranslator(client_key)
    error_occurred = False

    if 'deepl_trans' in df.columns:
        translations = df['deepl_trans'].dropna().tolist()
        start_idx = df['deepl_trans'].isnull().idxmax()
    else:
        translations = []
        start_idx = 0

    for text in tqdm(df['en'][start_idx:]):
        if error_occurred:
            print("Error Occured: DeepL")
            translations.extend([None] * (len(df['en']) - len(translations)))
            break

        try:
            translation = translator.translate_text(target_lang='KO', text=text)
        except:
            error_occurred = True
            translation = None
            
        translations.append(translation)

    df['deepl_trans'] = translations

    return df


def translate_dataset(df_path):
    """
    Translates a dataset containing English text to Korean using multiple translation services.

    Parameters:
    - df_path (str): Path to the CSV file containing the dataset.

    Returns:
    - df (DataFrame): Translated dataset with additional columns for each translation service.
    """
    df = pd.read_csv(df_path)

    print("[Papago Translation]")
    df = papago_translate(df)
    print("[Google Translation]")
    df = google_translate(df)
    print("[DeepL Translation]")
    df = deepl_translate(df)

    return df


if __name__ == '__main__':
    import argparse
    """
    [api_type]
    - google (No Authentication)
    - papago (ID & Secret)
    - deepl (Key)
    """
    # inference sentence
    parser = argparse.ArgumentParser(description='Inference Script')
    parser.add_argument(
        '--api_type', 
        type=str, 
        choices=['google', 'papago', 'deepl'],
        default='google', 
        help='Type of the model to use for inference'
    )
    parser.add_argument(
        '--en_text', 
        type=str, 
        default="NMIXX is a South Korean girl group that made a comeback on January 15, 2024 with their new song 'DASH'.",
        help='English text to be translated'
    )

    args = parser.parse_args()
    print(translate_sentence(args.en_text, api_type=args.api_type))
