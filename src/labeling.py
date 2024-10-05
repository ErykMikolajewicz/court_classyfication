import re

LABELS_REGEXES = {
    ".*_C_.*": "Cywilne pozew",
    ".*_Ns_.*": "Cywilne nieprocesowe",
    ".*_P_.*": "Prawo pracy",
    ".*_U_.*": "Ubezpieczenia społeczne",
    ".*Co_.*": "Inne cywilne",
}


def get_counter_label(counter_file_name: str) -> str:
    for regex, label in LABELS_REGEXES.items():
        regex_result = re.search(regex, counter_file_name)
        if regex_result is not None:
            return label

    return "Inne"  # If any label don't match
