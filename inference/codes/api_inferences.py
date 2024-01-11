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


def translate_single_text(text):
    """
    Translates a single text using multiple translation services.

    Parameters:
    - text (str): Text to be translated.

    Returns:
    - papago_translation (str): Translated text using Papago.
    - google_translation (str): Translated text using Google Translate.
    - deepl_translation (str): Translated text using DeepL.
    """
    # Papago
    papago_translator = PapagoTranslator(PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET)
    papago_translation = papago_translator.translate(src_lang='en', tgt_lang='ko', text=text)

    # Google
    google_translator = GoogleTranslator()
    google_translation = google_translator.translate(src='en', dest='ko', text=text).text

    # DeepL
    deepl_translator = DeeplTranslator(DEEPL_CLIENT_KEY)
    deepl_translation = deepl_translator.translate_text(target_lang='KO', text=text)

    return papago_translation, google_translation, deepl_translation


def papago_translate(df):
    translator = PapagoTranslator(PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET)
    error_occured = False

    translations = eval_df['papago_trans'].dropna().tolist()
    start_idx = eval_df['papago_trans'].isnull().idxmax()
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
    translator = GoogleTranslator()
    error_occured = False

    translations = []
    for text in tqdm(df['en']):
        if error_occured:
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


def deepl_translate(df):
    translator = DeeplTranslator(DEEPL_CLIENT_KEY)
    error_occured = False

    translations = []
    for text in tqdm(df['en']):
        if error_occured:
            print("Error Occured: DeepL")
            translations.extend([None] * (len(df['en']) - len(translations)))
            break

        try:
            translation = translator.translate_text(target_lang='KO', text=text)
        except:
            error_occurred = True
            translation = None
            
        translations.append(translation)

    df['deepl_trans'] = translation

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
    df = df[['source', 'en', 'ko']]

    df = papago_translate(df)
    df = google_translate(df)
    df = deepl_translate(df)

    return df


if __name__ == '__main__':
    # # single text inference
    # text = 'Python is great!'
    # papago, google, deepl = translate_single_text(text)
    # print("Papago:", papago)
    # print("Google:", google)
    # print("DeepL:", deepl)

    # # dataset inference
    # eval_path = './test_tiny_uniform100_inferenced.csv'
    # eval_df = translate_dataset(eval_path)
    # eval_df.to_csv('../results/test_tiny_uniform100_inferenced.csv', index=False)

    # papago continuous
    eval_path = '../results/test_tiny_uniform100_inferenced.csv'
    eval_df = pd.read_csv(eval_path)
    # start_idx = eval_df['papago_trans'].isnull().idxmax()
    # print(start_idx)
    eval_df = papago_translate(eval_df)
    eval_df.to_csv('../results/test_tiny_uniform100_inferenced.csv', index=False)
