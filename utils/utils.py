from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# klue_roberta-large_v01_detail.pt -> extract v"01"
def get_latest_version(full_model_name: str, model_save_dir: str, return_path: bool = False):
    model_save_dir = Path(model_save_dir)
    full_model_name = "_".join(full_model_name.split("/"))

    model_paths = list(model_save_dir.glob(f"{full_model_name}*.pt"))
    if len(model_paths) == 0:
        if return_path:
            raise FileNotFoundError(f"No model found in {model_save_dir} with name {full_model_name}")
        else:
            return 0

    model_paths = sorted(model_paths, key=lambda x: x.stem.split("_")[2][1:], reverse=True)

    latest_model_path = model_paths[0]
    latest_model_version = int(latest_model_path.stem.split("_")[2][1:])
    if return_path:
        return latest_model_path
    else:
        return latest_model_version


def fill_zeros(val: int, digit: int = 2) -> str:
    return str(val).zfill(digit)