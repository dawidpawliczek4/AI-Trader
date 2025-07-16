from pathlib import Path
from typing import Dict

import numpy as np
import evaluate
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import DataLoader
from ai_trader.nlp.data.load_data import make_dataloaders

def train_finbert(
    dataloaders: Dict[str, DataLoader],
    output_dir: str = "finbert-reg-v1",
    base_model: str = "ProsusAI/finbert",
    num_epochs: int = 3,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Fine-tune FinBERT as a regression model (sentiment 0-1).

    Args:
        dataloaders: dict with 'train', 'val' (and optionally 'test') DataLoader
        output_dir: where to save the best model + tokenizer
        base_model: HF checkpoint to start from
        num_epochs: training epochs
        lr: learning rate
        weight_decay: AdamW weight decay
        device: 'cuda' or 'cpu'

    Returns:
        Path to folder with best checkpoint
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=1,
        problem_type="regression",
        ignore_mismatched_sizes=True,
    ).to(device)

    train_ds = dataloaders["train"].dataset
    val_ds   = dataloaders["val"].dataset

    pearson = evaluate.load("pearsonr")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = preds.squeeze()
        return {
            "pearson": pearson.compute(predictions=preds, references=labels)["pearsonr"],
            "mae": float(np.abs(preds - labels).mean()),
        }

    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=lr,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=dataloaders["train"].batch_size,
        per_device_eval_batch_size=dataloaders["val"].batch_size,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="pearson",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print(f"Training FinBERT...")

    trainer.train()
    best_path = Path(trainer.state.best_model_checkpoint or output_dir)
    model.save_pretrained(best_path)
    tokenizer.save_pretrained(best_path)

    print(f"FinBERT saved to {best_path}")
    return best_path

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    dataloaders = make_dataloaders(tokenizer, batch_size=16, max_len=128)
    train_finbert(dataloaders, output_dir="finbert-reg-v1")
