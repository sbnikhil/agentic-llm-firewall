"""
DeBERTa Fine-Tuning Script — runs inside SageMaker container
=============================================================
Fine-tunes ProtectAI/deberta-v3-base-prompt-injection-v2 on our
combined injection dataset (LLMail-Inject + Enron negatives).

SageMaker passes data paths via environment variables:
    SM_CHANNEL_TRAIN  → directory containing train.csv
    SM_CHANNEL_VAL    → directory containing val.csv
    SM_MODEL_DIR      → where to save the fine-tuned model (copied to S3)

Do NOT run this locally — it is meant to run inside the SageMaker
HuggingFace training container.
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import wandb

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",    default="ProtectAI/deberta-v3-base-prompt-injection-v2")
    parser.add_argument("--epochs",        type=int,   default=3)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length",    type=int,   default=256)
    # SageMaker injects these
    parser.add_argument("--train",     default=os.environ.get("SM_CHANNEL_TRAIN", "data/finetune"))
    parser.add_argument("--val",       default=os.environ.get("SM_CHANNEL_VAL",   "data/finetune"))
    parser.add_argument("--model_dir", default=os.environ.get("SM_MODEL_DIR",     "model_output"))
    return parser.parse_args()


def load_data(train_dir, val_dir):
    train_df = pd.read_csv(Path(train_dir) / "train.csv").dropna()
    val_df   = pd.read_csv(Path(val_dir)   / "val.csv").dropna()

    # Reset index — Dataset.from_pandas() adds __index_level_0__ if index is
    # not reset, which causes the Trainer to error on unexpected input columns.
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    # Truncate very long texts to keep memory usage predictable
    train_df["text"] = train_df["text"].str[:1024]
    val_df["text"]   = val_df["text"].str[:1024]

    print(f"Train size : {len(train_df):,}  (pos={train_df['label'].sum():,})")
    print(f"Val size   : {len(val_df):,}  (pos={val_df['label'].sum():,})")
    return Dataset.from_pandas(train_df), Dataset.from_pandas(val_df)


def tokenize(dataset, tokenizer, max_length):
    return dataset.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        ),
        batched=True,
        remove_columns=["text"],
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    tp = ((predictions == 1) & (labels == 1)).sum()
    fp = ((predictions == 1) & (labels == 0)).sum()
    fn = ((predictions == 0) & (labels == 1)).sum()
    tn = ((predictions == 0) & (labels == 0)).sum()

    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1          = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    f2          = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
    fpr         = fp / (fp + tn) if (fp + tn) > 0 else 0
    accuracy    = (tp + tn) / len(labels)

    return {
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "f2":        round(f2,        4),
        "fpr":       round(fpr,       4),
    }


def main():
    args = parse_args()
    print(f"Starting fine-tune: {args.model_name}")
    print(f"Epochs={args.epochs}, batch={args.batch_size}, lr={args.learning_rate}")

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "llm-firewall-injection-detection"),
        config={
            "model":         args.model_name,
            "epochs":        args.epochs,
            "batch_size":    args.batch_size,
            "learning_rate": args.learning_rate,
            "max_length":    args.max_length,
            "dataset":       "LLMail-Inject-Phase1 + AstraDB-Enron",
        }
    )

    # Load data
    train_ds, val_ds = load_data(args.train, args.val)

    # Tokenizer + model — start from ProtectAI's already-specialized checkpoint
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    # Tokenize
    train_tok = tokenize(train_ds, tokenizer, args.max_length)
    val_tok   = tokenize(val_ds,   tokenizer, args.max_length)
    train_tok = train_tok.rename_column("label", "labels")
    val_tok   = val_tok.rename_column("label", "labels")
    train_tok.set_format("torch")
    val_tok.set_format("torch")

    # Training config
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f2",
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nStarting training...")
    trainer.train()

    # Save model + tokenizer
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    print(f"\nModel saved to: {model_dir}")

    # Final eval
    metrics = trainer.evaluate()
    print("\nFinal validation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
