import re

LABELS_REGEXES = {
    ".*_C_.*": "C",  # Cywilne pozew
    ".*_Ns_.*": "C",  # Cywilne nieprocesowe
    ".*_P_.*": "P",  # Prawo pracy
    ".*_U_.*": "U",  # Ubezpieczenia społeczne
    ".*Co_.*": "C",  # Inne cywilne
}


def get_counter_label(counter_file_name: str) -> str:
    for regex, label in LABELS_REGEXES.items():
        regex_result = re.search(regex, counter_file_name)
        if regex_result is not None:
            return label

    return "X"  # If any label don't match
