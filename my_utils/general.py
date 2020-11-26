
import numpy as np
import cv2
from pathlib import Path
import pandas as pd

from itertools import chain
from shutil import copyfile
from pprint import pprint
from tqdm.notebook import tqdm
from collections import defaultdict
from pdb import set_trace
from typing import *
from IPython.display import display


def ls(p, r='*'):
    return list(p.glob(r))

def flat(to_chain: list) -> list:
    return list(chain(*to_chain))

def rename_dict(dict_, rename_keys: dict):
    for old_key, new_key in rename_keys.items():
        dict_[new_key] = dict_.pop(old_key)
    return dict_
    
def len_dict(dict_):
    return {k: len(list_) for k, list_ in dict_.items()}

def is_key_match(dict_1, dict_2):
    dict_1_keys = pd.Series(list(dict_1.keys()))
    dict_2_keys = pd.Series(list(dict_2.keys()))
    return dict_1_keys.isin(dict_2_keys).all()

def mkdir_r(dir_path):
    for p in [*reversed((dir_path.parents)), dir_path]:
        if not p.is_dir():
            p.mkdir()

def pd_len_dict(dict_list: List[dict]) -> pd.DataFrame:
    display(pd.DataFrame([len_dict(d) for d in dict_list]))
    
