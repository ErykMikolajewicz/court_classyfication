"""Function to move counters into one dict, it make training model easier."""
import random
from itertools import chain
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from prefect import task


@task
def move_counters(court_type: str, test_fraction: float):
    counter_base_path = Path('data') / 'counters' / court_type
    counter_training_path = Path('data') / 'counters' / 'training'
    counter_validation_path = Path('data') / 'counters' / 'validation'

    counters_generators = []
    for appeal in counter_base_path.iterdir():
        for court in appeal.iterdir():
            counters_generators.append(court.iterdir())

    random.seed(42)
    counters_paths = list(chain(*counters_generators))
    random.shuffle(counters_paths)

    counters_number = len(counters_paths)

    training_part_length = int(counters_number * test_fraction)

    training_counters = counters_paths[:training_part_length]
    validation_counters = counters_paths[training_part_length:]

    def save_counter(set_path: Path, counter_path: Path):
        with open(counter_path, 'rb') as file:
            counter = file.read()
        destination_path = set_path / counter_path.name
        with open(destination_path, 'wb') as file:
            file.write(counter)

    with ThreadPoolExecutor(max_workers=14) as executor:
        save_counter_training = partial(save_counter, counter_training_path)
        executor.map(save_counter_training, training_counters)

        save_counter_validation = partial(save_counter, counter_validation_path)
        executor.map(save_counter_validation, validation_counters)