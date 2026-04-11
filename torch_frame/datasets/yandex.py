from __future__ import annotations

import os.path as osp
import os
import zipfile
from typing import Any

import numpy as np
import pandas as pd
import json
import torch_frame
from torch_frame.utils.split import SPLIT_TO_NUM
# SPLIT_TO_NUM = {'train': 0, 'val': 1, 'test': 2}

SPLIT_COL = 'split_col'
TARGET_COL = 'target'




def col2stype(df):
    col_to_stype: dict[str, torch_frame.stype] = {}
    c_col_names=[]
    n_col_names=[]
    for col_name in df.columns:
        if col_name == TARGET_COL:
            continue
        if col_name.startswith('C_feature'):
            col_to_stype[col_name] = torch_frame.categorical
            c_col_names.append(col_name)
        elif col_name.startswith('N_feature'):
            col_to_stype[col_name] = torch_frame.numerical
            n_col_names.append(col_name)

    if n_col_names is not None:
        for n_col in n_col_names:
            df[n_col] = df[n_col].astype('float64')
    return df, col_to_stype



class Yandex(torch_frame.data.Dataset):
    def __init__(self, train_df, val_df, test_df, name: str, task_type: str) -> None:
        # 標記split_col
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        train_df['split_col'] = 0
        val_df['split_col'] = 1
        test_df['split_col'] = 2
        # 合併
        df = pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True)
        self.df, col_to_stype = col2stype(df)
        if task_type == "regression":
            col_to_stype[TARGET_COL] = torch_frame.numerical
        else:
            col_to_stype[TARGET_COL] = torch_frame.categorical
        self.name = name
        super().__init__(self.df, col_to_stype, target_col=TARGET_COL, split_col=SPLIT_COL)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(name='{self.name}')")