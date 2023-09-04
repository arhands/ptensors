from __future__ import annotations
import torch
from torch import Tensor
from typing import Literal, TypeAlias, Union, List, Tuple, Dict
import os

def get_run_path(base_dir: str = 'logs') -> str:
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
        run_id = 0
    else:
        contents = os.listdir(base_dir)
        contents = [int(c) for c in contents]
        run_id = 1 + max(*contents) if len(contents) > 1 else 1 + contents[0]
    run_path = f"{base_dir}/{run_id}/"
    os.mkdir(run_path)
    return run_path