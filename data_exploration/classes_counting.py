"""Make chart with information about number of classes, also get some basic info, about words counts from data."""
from pathlib import Path
import re
from collections import Counter
import pickle
from itertools import chain
from typing import Literal

import matplotlib.pyplot as plt

from src.labeling import LABELS_REGEXES


def plot_classes_chart(court_type: str, label_type: Literal["detailed", "general"]):
    label_counter = Counter()
    general_world_counter = Counter()

    counter_dir = Path('data') / 'counters' / court_type

    cases = []
    for appeal in counter_dir.iterdir():
        for court in appeal.iterdir():
            cases.append(court.iterdir())
    cases_counters = chain(*cases)

    for counter_path in cases_counters:
        with open(counter_path, 'rb') as counter_file:
            case_counter = pickle.load(counter_file)
        general_world_counter.update(case_counter)

        for regex, label in LABELS_REGEXES.items():
            label = label[label_type]
            if regex == '.*':
                print(counter_path)
            regex_result = re.search(regex, counter_path.name)
            if regex_result:
                label_counter[label] += 1
                break


    print(f'Liczba słów: {len(general_world_counter)}')

    words = general_world_counter.most_common()
    most_popular_words = words[:50]
    least_popular_words = words[-50:]
    print('Najpopularniejsze słowa:', most_popular_words)
    print('Najmniej popularne słowa:', least_popular_words)

    sorted_labels_and_counts = label_counter.most_common()

    total = sum(label_counter.values())

    labels = [f'{label} {count*100/total:2.2f}% ({count})' for label, count in sorted_labels_and_counts]

    # Legend config is strange, but work correctly. Legend is visible and do not cover chart
    plt.legend(labels=labels, bbox_to_anchor=(0.1, 0.7), loc='right')

    plt.title('Procentowy udział według typów spraw')

    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()

    plt.show()
