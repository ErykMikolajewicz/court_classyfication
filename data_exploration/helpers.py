from pathlib import Path
import re
import os

import tiktoken


def get_documents_length(path_to_documents: Path) -> list[int]:
    documents_length = []
    for path in path_to_documents.iterdir():
        if path.is_dir():
            subfolders_documents_lengths = get_documents_length(path)
            documents_length.extend(subfolders_documents_lengths)
        else:
            with open(path, 'r') as file:
                document_length = len(file.read())
            documents_length.append(document_length)
    return documents_length


def get_documents_tokens_number(path_to_documents: Path) -> list[int]:
    threads_number = os.cpu_count()
    encoding = tiktoken.get_encoding('o200k_base')
    documents_tokens_lengths = []
    files_str = []
    for path in path_to_documents.iterdir():
        if path.is_dir():
            subfolders_documents_lengths = get_documents_tokens_number(path)
            documents_tokens_lengths.extend(subfolders_documents_lengths)
        else:
            with open(path, 'r') as file:
                file_str = file.read()
            files_str.append(file_str)
    files_tokens = encoding.encode_batch(files_str, num_threads=threads_number)
    files_tokens_lengths = [len(tokens) for tokens in files_tokens]
    documents_tokens_lengths.extend(files_tokens_lengths)
    return documents_tokens_lengths


def seek_case_k_signature(case_text):
    result = re.search(r'K [0-9]+/\d{2}', case_text)
    return result
