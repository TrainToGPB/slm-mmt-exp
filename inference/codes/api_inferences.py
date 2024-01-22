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
    """
    Translate English text to Korean using the Papago translation service and update the DataFrame.

    Parameters:
    - df (DataFrame): DataFrame containing the 'en' column with English text to be translated.

    Returns:
    - df (DataFrame): Updated DataFrame with the 'papago_trans' column containing Korean translations.
    """
    translator = PapagoTranslator(PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET)
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

    translations = []
    for text in tqdm(df['en']):
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


def deepl_translate(df):
    """
    Translate English text to Korean using the DeepL translation service and update the DataFrame.

    Parameters:
    - df (DataFrame): DataFrame containing the 'en' column with English text to be translated.

    Returns:
    - df (DataFrame): Updated DataFrame with the 'deepl_trans' column containing Korean translations.
    """
    translator = DeeplTranslator(DEEPL_CLIENT_KEY)
    error_occurred = False

    translations = []
    for text in tqdm(df['en']):
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
    """
    [EVAL_PATH]
    AI Hub Integrated Uniform 100 (Total 800): ../../translation_datasets/aihub_integration/uniform_for_evaluation/test_tiny_uniform100.csv
    Flores-101 (Total 1012): ../../translation_datasets/flores_101/test_flores.csv

    [SAVE_PATH]
    AI Hub Integrated Uniform 100 (Total 800, also continuous eval path for papago): ../results/test_tiny_uniform100_inferenced.csv
    Flores-101 (Total 1012, also continuous eval path for papago): ../results/test_flores_inferenced.csv
    """
    # # dataset inference
    # eval_path = '../../translation_datasets/flores_101/test_flores.csv'
    # save_path = '../results/test_flores_inferenced.csv'
    # eval_df = translate_dataset(save_path)
    # eval_df.to_csv(save_path, index=False)

    # # papago continuous inference
    # eval_path = '../results/test_tiny_uniform100_inferenced.csv'
    # eval_df = pd.read_csv(eval_path)
    # eval_df = papago_translate(eval_df)
    # eval_df.to_csv('../results/test_tiny_uniform100_inferenced.csv', index=False)

    # inference again...
    eval_path = '../../translation_datasets/flores_101/test_flores.csv'
    save_path = '../results/test_flores_inferenced.csv'
    eval_df = pd.read_csv(eval_path)
    eval_df = google_translate(eval_df)
    eval_df.to_csv(save_path, index=False)
