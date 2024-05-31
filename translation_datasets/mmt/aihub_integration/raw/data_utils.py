import os
from zipfile import ZipFile

import chardet
from unidecode import unidecode


def extract_all_zips(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files: 
            if file.endswith('.zip'):
                zipfile_path = os.path.join(root, file)
                
                with ZipFile(zipfile_path, 'r') as zip_ref:
                    zip_ref.extractall(root)

                os.remove(zipfile_path)
                print(f"Extracted and Removed: {zipfile_path}")


def detect_encoding(file_name):
    with open(file_name, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    return result['encoding']


def decode_file_name(file_name, encodings=['cp949', 'euc-kr', 'utf-8']):
    for encoding in encodings:
        try:
            decoded_name = file_name.encode(encoding, 'ignore').decode('utf-8', 'ignore')
            return decoded_name
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return file_name


def rename_files_in_directory(directory, encodings=['cp949', 'euc-kr', 'utf-8']):
    for root, dirs, files in os.walk(directory, topdown=False):
        
        for name in files:
            old_path = os.path.join(root, name)
            new_name = decode_file_name(name, encodings)
            new_path = os.path.join(root, new_name)
            if old_path != new_path:
                try:
                    os.rename(old_path, new_path)
                except OSError as e:
                    print(f"Error renaming file {old_path} to {new_path}: {e}")

        for name in dirs:
            old_path = os.path.join(root, name)
            new_name = decode_file_name(name, encodings)
            new_path = os.path.join(root, new_name)
            if old_path != new_path:
                try:
                    os.rename(old_path, new_path)
                except OSError as e:
                    print(f"Error renaming directory {old_path} to {new_path}: {e}")


if __name__ == "__main__":
    folder_path = './'
    # extract_all_zips(folder_path)
    rename_files_in_directory(folder_path, ['cp949'])
