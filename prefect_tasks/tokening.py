import pickle
from pathlib import Path
import random
from types import SimpleNamespace

from prefect import task
import tiktoken


@task
def create_tokens(court_type: str, appeal_name: str, court_name: str, fraction: float):
    justification_base_path = Path('data') / 'justification'
    tokens_training_path = Path('data') / 'tokens' / 'training'
    tokens_validation_path = Path('data') / 'tokens' / 'validation'

    encoding = tiktoken.get_encoding('o200k_base')

    justification_particular_dir = justification_base_path / court_type / appeal_name / court_name

    arrays_to_save = []
    for file_path in justification_particular_dir.iterdir():
        file_path: Path
        text = file_path.read_text()
        tokens_array = encoding.encode_to_numpy(text)
        saving_data = SimpleNamespace(label=file_path.stem, array=tokens_array)
        arrays_to_save.append(saving_data)

    encoded_files_length = len(arrays_to_save)

    random.seed(42)
    random.shuffle(arrays_to_save)

    test_length = int(encoded_files_length * fraction)
    save_to_test = arrays_to_save[:test_length]
    save_to_validation = arrays_to_save[test_length:]

    for saving_info in save_to_test:
        label = saving_info.label
        array = saving_info.array
        save_path = tokens_training_path / (label + '.pickle')
        with open(save_path, 'wb') as file:
            pickle.dump(array, file)

    for saving_info in save_to_validation:
        label = saving_info.label
        array = saving_info.array
        save_path = tokens_validation_path / (label + '.pickle')
        with open(save_path, 'wb') as file:
            pickle.dump(array, file)
