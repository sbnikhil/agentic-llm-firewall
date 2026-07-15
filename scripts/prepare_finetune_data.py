import random
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

BUCKET = os.getenv("S3_BUCKET", "llm-firewall")
S3_PREFIX = "finetune"
POSITIVE_SAMPLE = 200_000  # capped by actual unique count (~153K)
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def load_positives():
    """LLMail-Inject Phase1 body — real email-format injection attempts."""
    from datasets import load_dataset
    print("Loading LLMail-Inject Phase1 (positives)...")
    ds = load_dataset("microsoft/llmail-inject-challenge", split="Phase1")
    bodies = list(set(row["body"] for row in ds if row.get("body", "").strip()))
    sampled = random.sample(bodies, min(POSITIVE_SAMPLE, len(bodies)))
    print(f"  Unique bodies    : {len(bodies):,}")
    print(f"  Sampled          : {len(sampled):,}")
    return sampled


def load_negatives(target_count: int):
    """
    Legitimate text from two sources:
      1. deepset/prompt-injections label=0 (train split)
      2. AstraDB company_knowledge collection — our actual production documents

    AstraDB is preferred over re-downloading raw HuggingFace data because:
      - Documents are already quality-filtered (30-word minimum, noise removed)
      - These are exactly what the model sees at inference time
      - More realistic negatives = better training signal
    """
    from datasets import load_dataset
    from astrapy import DataAPIClient

    negatives = []

    # Source 1: deepset legitimate queries (short, query-style text)
    print("Loading deepset negatives (train split)...")
    ds_deepset = load_dataset("deepset/prompt-injections", split="train")
    deepset_neg = [row["text"] for row in ds_deepset if row["label"] == 0 and row["text"].strip()]
    negatives.extend(deepset_neg)
    print(f"  deepset label=0: {len(deepset_neg):,}")

    # Source 2: AstraDB production documents (quality-filtered Enron emails)
    # These are the actual documents our system retrieves — the model should
    # learn to distinguish attacks from exactly this content.
    print("Loading negatives from AstraDB (production knowledge base)...")
    needed = target_count - len(negatives)
    try:
        client = DataAPIClient(os.getenv("ASTRA_DB_APPLICATION_TOKEN"))
        db = client.get_database_by_api_endpoint(
            os.getenv("ASTRA_DB_API_ENDPOINT"),
            keyspace=os.getenv("ASTRA_DB_KEYSPACE", "logs"),
        )
        collection = db.get_collection("company_knowledge")

        astra_docs = []
        PAGE_SIZE = 20  # AstraDB's native page size — keeps each request short
        retries = 0
        max_retries = 3

        while len(astra_docs) < needed and retries < max_retries:
            try:
                cursor = collection.find(
                    {},
                    projection={"content": True},
                )
                for doc in cursor:
                    text = doc.get("content", "").strip()
                    if text and len(text.split()) >= 10:
                        astra_docs.append(text)
                    if len(astra_docs) >= needed:
                        break
                break  # success — exit retry loop
            except Exception as e:
                retries += 1
                print(f"  AstraDB read interrupted ({e}), retry {retries}/{max_retries}...")
                time.sleep(5)

        negatives.extend(astra_docs)
        print(f"  AstraDB documents: {len(astra_docs):,}")
    except Exception as e:
        print(f"  AstraDB query failed: {e}")
        print("  Falling back to HuggingFace Enron dataset...")
        from datasets import load_dataset as lds
        ds_enron = lds("corbt/enron-emails", split="train")
        enron_bodies = [
            row["body"] for row in ds_enron
            if row.get("body", "").strip() and len(row["body"].split()) >= 30
        ]
        enron_sample = random.sample(enron_bodies, min(needed, len(enron_bodies)))
        negatives.extend(enron_sample)
        print(f"  Enron fallback: {len(enron_sample):,}")

    print(f"  Total negatives: {len(negatives):,}")
    return negatives[:target_count]


def build_dataset(positives, negatives):
    """Combine, shuffle, split 80/20."""
    data = (
        [{"text": t, "label": 1} for t in positives] +
        [{"text": t, "label": 0} for t in negatives]
    )
    random.shuffle(data)

    split = int(len(data) * 0.8)
    train = data[:split]
    val   = data[split:]
    print(f"\nDataset split:")
    print(f"  Train : {len(train):,}")
    print(f"  Val   : {len(val):,}")
    return train, val


def save_csv(data, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(data)


def upload_to_s3(local_path, s3_key):
    import boto3
    s3 = boto3.client("s3")
    print(f"Uploading {local_path} → s3://{BUCKET}/{s3_key}")
    s3.upload_file(str(local_path), BUCKET, s3_key)
    print(f"  Done.")


def main():
    print("=" * 55)
    print("  FINE-TUNE DATASET PREPARATION")
    print("=" * 55)

    positives = load_positives()
    negatives = load_negatives(target_count=len(positives))
    train, val = build_dataset(positives, negatives)

    # Save locally
    data_dir = Path(__file__).parent.parent / "data" / "finetune"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    val_path   = data_dir / "val.csv"
    save_csv(train, train_path)
    save_csv(val,   val_path)
    print(f"\nSaved locally:")
    print(f"  {train_path}")
    print(f"  {val_path}")

    # Upload to S3
    print()
    upload_to_s3(train_path, f"{S3_PREFIX}/train.csv")
    upload_to_s3(val_path,   f"{S3_PREFIX}/val.csv")

    print(f"\nData ready at s3://{BUCKET}/{S3_PREFIX}/")
    print("Next: python scripts/run_sagemaker_job.py")


if __name__ == "__main__":
    main()
