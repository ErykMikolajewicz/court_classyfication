from pathlib import Path
import re
from collections import Counter
import pickle

import matplotlib.pyplot as plt

from src.labeling import LABELS_REGEXES

label_counter = Counter()
general_world_counter = Counter()

data_path = Path('data') / 'training' / 'counters'
for counter_path in data_path.iterdir():
    with open(counter_path, 'rb') as counter_file:
        case_counter = pickle.load(counter_file)
    general_world_counter.update(case_counter)

    is_other = True
    for regex, label in LABELS_REGEXES.items():
        regex_result = re.search(regex, counter_path.name)
        if regex_result:
            label_counter[label] += 1
            is_other = False
            break
    if is_other:
        label_counter['Inne'] += 1

print(f'Liczba słów: {len(general_world_counter)}')

words = general_world_counter.most_common()
most_popular_words = words[:50]
least_popular_words = words[-50:]
print('Najpopularniejsze słowa:', most_popular_words)
print('Najmniej popularne słowa:', least_popular_words)

sorted_labels_and_counts = label_counter.most_common()

total = sum(label_counter.values())

labels = [f'{label} {count*100/total:2.2f}% ({count})' for label, count in sorted_labels_and_counts]

patches = plt.pie([count for _, count in sorted_labels_and_counts])
plt.legend(labels=labels, bbox_to_anchor=(0.8, 0.8))

# Maximize window during show
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())

plt.title('Procentowy udział według typów spraw')
plt.show()
