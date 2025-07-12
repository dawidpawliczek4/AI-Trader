import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from typing import Dict, Optional

class HeadlineDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int = 128) -> None:
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = self.labels[idx]
        return item


def make_dataloaders(
    tokenizer,
    csv_path: Optional[str] = None,
    batch_size: int = 32,
    max_len: int = 128,
    random_state: int = 42,
):
    """
    Create PyTorch DataLoaders for train/val/test from a CSV with
    'Headline' and 'Sentiment' columns (sentiment in range 0-1).

    Args:
        tokenizer: Hugging Face tokenizer (e.g. AutoTokenizer.from_pretrained(...))
        csv_path: path to CSV; if None → '../dataset.csv'
        batch_size: batch size
        max_len: max token length
        random_state: train/test split seed

    Returns:
        dict with keys 'train', 'val', 'test' -> DataLoader
    """
    if csv_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(os.path.dirname(current_dir), "dataset.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    df = pd.read_csv(csv_path)
    if {"Headline", "Sentiment"} - set(df.columns):
        raise ValueError("CSV must contain 'Headline' and 'Sentiment' columns")

    # Split 80 / 10 / 10
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df["Headline"].tolist(),
        df["Sentiment"].values,
        test_size=0.2,
        random_state=random_state,
        shuffle=True,
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=0.5,
        random_state=random_state,
        shuffle=True,
    )

    def make_loader(texts, labels, shuffle=False):
        ds = HeadlineDataset(texts, labels, tokenizer, max_len)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return {
        "train": make_loader(train_texts, train_labels, shuffle=True),
        "val": make_loader(val_texts, val_labels),
        "test": make_loader(test_texts, test_labels),
    }
