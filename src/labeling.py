import re
from typing import Literal
from pathlib import Path
import json

import xattr


LABELS_REGEXES = {
    ".*_C_.*": {"detailed": "Cywilne pozew", "general": "Cywilne"},
    ".*_Ca_.*": {"detailed": "Cywilne pozew apelacja w okręgowym", "general": "Cywilne"},
    ".*Co_.*": {"detailed": "Inne cywilne", "general": "Cywilne"},
    ".*_Cz_.*": {"detailed": "Cywilne zażalenie w okręgowym", "general": "Cywilne"},
    ".*_Ga_.*": {"detailed": "Gospodarcze apelacja w okręgowym", "general": "Gospodarcze"},
    ".*_GC_.*": {"detailed": "Gospodarcze inne", "general": "Gospodarcze"},
    ".*_Gz_.*": {"detailed": "Gospodarcze zażalenie w okręgowym", "general": "Gospodarcze"},
    ".*_K_.*": {"detailed": "Karne", "general": "Karne"},
    ".*_Ka_.*": {"detailed": "Karne apelacja w okręgowym", "general": "Karne"},
    ".*_Ko_.*": {"detailed": "Karne z urzędu", "general": "Karne"},
    ".*_Kop_.*": {"detailed": "Karne ze stosunków międzynarodowych", "general": "Karne"},
    ".*_Kow_.*": {"detailed": "Karne kwestie więźniów", "general": "Karne"},
    ".*_Kp_.*": {"detailed": "Karne przygotowawcze", "general": "Karne"},
    ".*_Kz_.*": {"detailed": "Karne zażalenie w okręgowym", "general": "Karne"},
    ".*_Kzw_*": {"detailed": "Karne wykonawcze zażalenie w okręgowym", "general": "Karne"},
    ".*_Ns_.*": {"detailed": "Cywilne nieprocesowe", "general": "Cywilne"},
    ".*_Nc_.*": {"detailed": "Cywilne nakazowe i upominawcze", "general": "Cywilne"},
    ".*_P_.*": {"detailed": "Prawo pracy", "general": "Pracy"},
    ".*_Pa_.*": {"detailed": "Prawo pracy apelacja w okręgowym", "general": "Pracy"},
    ".*_Po_.*": {"detailed": "Prawo pracy inne", "general": "Pracy"},
    ".*_Pz_.*": {"detailed": "Prawo pracy zażalenie w okręgowym", "general": "Pracy"},
    ".*_S_.*": {"detailed": "Skargi", "general": "Skargi"},
    ".*_U_.*": {"detailed": "Ubezpieczenia społeczne", "general": "Ubezpieczenia Społeczne"},
    ".*_Uo_.*": {"detailed": "Ubezpieczenia społeczne inne", "general": "Ubezpieczenia Społeczne"},
    ".*_Ua_.*": {"detailed": "Ubezpieczenia społeczne apelacja w okręgowym", "general": "Ubezpieczenia Społeczne"},
    ".*_Uz_.*_Uz_.*": {"detailed": "Ubezpieczenia społeczne zażalenie w okręgowym",
                       "general": "Ubezpieczenia Społeczne"},
    ".*_Zs_.*": {"detailed": "Skarga postanowienie krajowej izby odwoławczej", "general": "Skargi"},
    ".*": {"detailed": "Inne", "general": "Inne"}
}


counters_dir = Path('data') / 'counters'
def get_counter_label(file_name: str, label_type: Literal['detailed', 'general', 'appeal']) -> str:
    if label_type in ('detailed', 'general'):
        for regex, label in LABELS_REGEXES.items():
            regex_result = re.search(regex, file_name)
            if regex_result is not None:
                return label[label_type]
        else: # to ide do not complain about invalid type hints
            raise  ValueError
    elif label_type == 'appeal':
        counter_path = counters_dir / file_name
        attributes = xattr.getxattr(counter_path, attr='user.attributes')
        attributes = json.loads(attributes)
        return attributes[label_type]
    else:
        raise ValueError

appeals = [ "wroclawska",
"katowicka",
"krakowska",
"rzeszowska",
"poznanska",
"lodzka",
"lubelska",
"warszawska",
"szczecinska",
"gdanska",
"bialostocka"
]

def get_labels(label_type: Literal['detailed', 'general', 'appeal']) -> list:
    if label_type in ('detailed', 'general'):
        return list({label[label_type] for label in LABELS_REGEXES.values()})
    elif label_type == 'appeal':
        return appeals
    else:
        raise ValueError
