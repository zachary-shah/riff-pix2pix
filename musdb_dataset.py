"""
Adapted from https://github.com/timothybrooks/instruct-pix2pix/blob/main/edit_dataset.py
"""

import random

random.seed(0)
import os
import json
from pathlib import Path
from random import shuffle

import torch
from PIL import Image
from torch.utils.data import Dataset


"""
Dataset for loading in Musdb18 spectrograms (processed with preprocess_musdb.py) for upload preparation

Dataset assumes data located at rootdir with the following:
<rootdir>
    metadata.jsonl
    input_image/
        img1.png
        ...
        imgn.png
    edited_image/
        img1.png
        ...
        imgn.png

metadata.jsonl contains all edit prompts for each image pair, as well as the filepaths relative to rootdir for 
the input and edited images.

Parameters
----------
rootdir : str
    Root to training data directory as described above
return_paths : bool (optional)
    If return_paths == True, returns filepaths instead of images. Otherwise, returns full images
num_samples_to_use : int (optional)
    caps the number of samples to generate in dataset (by random sample)
"""
class MusdbDataset(Dataset):
    def __init__(self, rootdir: str, num_samples_to_use: int = None, return_paths: bool = True):

        self.data = []
        self.rootdir = rootdir
        self.return_paths = return_paths

        with open(os.path.join(rootdir, "metadata.jsonl"), 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        if num_samples_to_use is not None:
            self.data = random.sample(self.data, len(self.data))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> dict:

        item = self.data[i]

        input_file_path = os.path.join(self.rootdir,item["input_file_path"])
        edited_file_path = os.path.join(self.rootdir,item["edited_file_path"])
        edit_prompt = item["edit_prompt"]

        if self.return_paths:
            return dict(
                original_image=input_file_path,
                original_prompt=edit_prompt,
                edit_prompt=edit_prompt,
                edited_prompt=edit_prompt,
                edited_image=edited_file_path,
            )

        input_image = Image.open(input_file_path).convert("RGB")
        edited_image = Image.open(edited_file_path).convert("RGB")

        return dict(
            original_image=input_image,
            original_prompt=edit_prompt,
            edit_prompt=edit_prompt,
            edited_prompt=edit_prompt,
            edited_image=edited_image,
        )