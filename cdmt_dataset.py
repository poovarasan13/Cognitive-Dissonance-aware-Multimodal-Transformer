# cdmt_dataset.py

import os
from typing import Dict, Any, List

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizerFast


class MemeDataset(Dataset):
    """
    Generic meme dataset for Memotion 2.0 / Hateful Memes style CSV.
    """

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        label_map: Dict[str, int],
        max_length: int = 48,
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.label_map = label_map
        self.max_length = max_length

        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        img_path = os.path.join(self.image_root, str(row["image_path"]))
        text = str(row["text"])
        label_str = str(row["label"])
        label = self.label_map[label_str]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        sample = {
            "images": image,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        return sample


def build_label_map(labels: List[str]) -> Dict[str, int]:
    return {name: i for i, name in enumerate(labels)}
