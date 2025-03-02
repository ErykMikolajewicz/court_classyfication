import re

LABELS_REGEXES = {
    ".*_C_.*": "Cywilne pozew",
    ".*_Ca_.*": "Cywilne pozew apelacja w okręgowym",
    ".*Co_.*": "Inne cywilne",
    ".*_Cz_.*": "Cywilne zażalenie w okręgowym",
    ".*_Ga_.*": "Gospodarcze apelacja w okręgowym",
    ".*_GC_.*": "Gospodarcze inne",
    ".*_Gz_.*": "Gospodarcze zażalenie w okręgowym",
    ".*_K_.*": "Karne",
    ".*_Ka_.*": "Karne apelacja w okręgowym",
    ".*_Kz_.*": "Karne zażalenie w okręgowym",
    ".*_Ko_.*": "Karne z urzędu",
    ".*_Ns_.*": "Cywilne nieprocesowe",
    ".*_P_.*": "Prawo pracy",
    ".*_Pa_.*": "Prawo pracy apelacja w okręgowym",
    ".*_Pz_.*": "Prawo pracy zażalenie w okręgowym",
    ".*_S_.*": "Skargi",
    ".*_U_.*": "Ubezpieczenia społeczne",
    ".*_Ua_.*": "Ubezpieczenia społeczne apelacja w okręgowym",
    ".*": "Inne"
    #".*_Uz_.*": "Ubezpieczenia społeczne zażalenie w okręgowym"
}


def get_counter_label(counter_file_name: str) -> str:
    for regex, label in LABELS_REGEXES.items():
        regex_result = re.search(regex, counter_file_name)
        if regex_result is not None:
            return label
    else: # to ide do not complain about invalid type hints
        raise  ValueError
