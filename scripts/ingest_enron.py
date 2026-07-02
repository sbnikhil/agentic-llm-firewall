"""
Enron Email Corpus Ingestion Pipeline
======================================
Loads the Enron email dataset from HuggingFace, filters for quality,
assigns RBAC access levels by organizational hierarchy, and ingests
into AstraDB with batch processing and checkpointing.

Dataset: corbt/enron-emails (~517K emails)
After quality filtering: ~300K emails expected

Access level assignment (derived from sender role, not manually labeled):
  - executive senders  → confidential
  - manager/director   → internal
  - all others         → public

Usage:
    python scripts/ingest_enron.py

Resumable: progress is checkpointed to ingest_checkpoint.json.
Delete that file to restart from scratch.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ingest_enron.log")
    ]
)
logger = logging.getLogger(__name__)

CHECKPOINT_FILE = Path("ingest_checkpoint.json")
BATCH_SIZE = 64          # Embeddings per batch
ASTRA_BATCH_SIZE = 20    # Documents per AstraDB write (API limit)
MIN_WORD_COUNT = 30      # Filter out very short emails
COLLECTION_NAME = "company_knowledge"

# ── Enron executive roster (well-documented, used for tier labeling) ──────────
EXECUTIVE_EMAILS = {
    "kenneth.lay", "ken.lay", "jeffrey.skilling", "jeff.skilling",
    "andrew.fastow", "andy.fastow", "rebecca.mark", "lou.pai",
    "richard.causey", "rick.causey", "mark.frevert", "greg.whalley",
    "john.lavorato", "louise.kitchen", "david.delainey", "ray.bowen",
    "stanley.horton", "james.derrick", "rex.shelby", "ben.glisan",
    "michael.kopper", "richard.buy", "sally.beck", "mark.koenig"
}

MANAGER_KEYWORDS = {
    "vp", "vice president", "director", "manager", "head", "chief",
    "president", "supervisor", "lead", "principal", "senior"
}


def get_access_level(sender: str, message_body: str) -> str:
    """
    Derive RBAC access level from sender role — not manual labeling.
    Mirrors how real enterprise systems assign sensitivity by org hierarchy.
    """
    if not sender:
        return "public"

    sender_lower = sender.lower()
    local_part = sender_lower.split("@")[0].replace(".", " ").replace("_", " ")

    if any(exec_name in sender_lower for exec_name in EXECUTIVE_EMAILS):
        return "confidential"

    first_line = message_body[:200].lower() if message_body else ""
    if any(kw in local_part or kw in first_line for kw in MANAGER_KEYWORDS):
        return "internal"

    return "public"


def is_quality_email(text: str, subject: str) -> bool:
    """
    Filter out auto-replies, calendar invites, and noise.
    Production systems filter before indexing to improve retrieval signal.
    """
    if not text or len(text.split()) < MIN_WORD_COUNT:
        return False

    noise_subjects = [
        "auto:", "automatic reply", "out of office", "ooo:", "vacation",
        "delivery failure", "undeliverable", "meeting request",
        "accepted:", "declined:", "tentative:", "fw:", "fwd:"
    ]
    subject_lower = (subject or "").lower()
    if any(noise in subject_lower for noise in noise_subjects):
        return False

    noise_starters = [
        "this message is a reminder", "please join", "-----original message",
        "this email and any attachments", "confidentiality notice"
    ]
    text_lower = text[:100].lower()
    if any(starter in text_lower for starter in noise_starters):
        return False

    return True


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"processed": 0, "ingested": 0, "skipped": 0, "by_level": {"public": 0, "internal": 0, "confidential": 0}}


def save_checkpoint(state: dict):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(state, f, indent=2)


def ingest():
    logger.info("=" * 60)
    logger.info("ENRON CORPUS INGESTION PIPELINE")
    logger.info("=" * 60)

    # ── Connect to AstraDB ────────────────────────────────────────────────────
    logger.info("Connecting to AstraDB...")
    try:
        from langchain_astradb import AstraDBVectorStore
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_core.documents import Document

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": BATCH_SIZE, "normalize_embeddings": True}
        )

        vstore = AstraDBVectorStore(
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
            token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
        )
        logger.info("AstraDB connection successful.")
    except Exception as e:
        logger.error(f"AstraDB connection failed: {e}")
        sys.exit(1)

    # ── Load dataset ──────────────────────────────────────────────────────────
    logger.info("Loading Enron dataset from HuggingFace (corbt/enron-emails)...")
    logger.info("This downloads ~500MB on first run and is cached locally after.")
    try:
        from datasets import load_dataset
        dataset = load_dataset("corbt/enron-emails", split="train")
        logger.info(f"Dataset loaded: {len(dataset):,} total emails")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    state = load_checkpoint()
    start_idx = state["processed"]
    if start_idx > 0:
        logger.info(f"Resuming from checkpoint: {start_idx:,} already processed")

    # ── Process and ingest ────────────────────────────────────────────────────
    batch: list[Document] = []
    t_start = time.time()

    for i, row in enumerate(dataset):
        if i < start_idx:
            continue

        text = row.get("body", "") or ""
        subject = row.get("subject", "") or ""
        sender = row.get("from", "") or ""

        # Quality filter
        if not is_quality_email(text, subject):
            state["skipped"] += 1
            state["processed"] += 1
            continue

        access_level = get_access_level(sender, text)

        # Truncate very long emails — keep the first 1500 words
        words = text.split()
        if len(words) > 1500:
            text = " ".join(words[:1500])

        doc = Document(
            page_content=f"Subject: {subject}\n\n{text}",
            metadata={
                "access_level": access_level,
                "source": "enron_corpus",
                "sender": sender[:100] if sender else "unknown",
                "date": row.get("date", "") or ""
            }
        )

        batch.append(doc)
        state["by_level"][access_level] += 1
        state["processed"] += 1

        # Ingest batch
        if len(batch) >= ASTRA_BATCH_SIZE:
            try:
                vstore.add_documents(batch)
                state["ingested"] += len(batch)
                batch = []
                save_checkpoint(state)

                elapsed = time.time() - t_start
                rate = state["ingested"] / elapsed if elapsed > 0 else 0
                remaining = (len(dataset) - i) / rate if rate > 0 else 0

                logger.info(
                    f"Ingested {state['ingested']:,} | "
                    f"Skipped {state['skipped']:,} | "
                    f"Rate {rate:.0f}/s | "
                    f"ETA {remaining/60:.1f}min | "
                    f"Public {state['by_level']['public']:,} | "
                    f"Internal {state['by_level']['internal']:,} | "
                    f"Confidential {state['by_level']['confidential']:,}"
                )
            except Exception as e:
                logger.warning(f"Batch write failed at index {i}: {e}. Retrying after 5s...")
                time.sleep(5)
                try:
                    vstore.add_documents(batch)
                    state["ingested"] += len(batch)
                    batch = []
                except Exception as e2:
                    logger.error(f"Retry failed: {e2}. Checkpoint saved, restart to resume.")
                    save_checkpoint(state)
                    sys.exit(1)

    # Flush remaining batch
    if batch:
        vstore.add_documents(batch)
        state["ingested"] += len(batch)
        save_checkpoint(state)

    # ── Final report ──────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total processed : {state['processed']:,}")
    logger.info(f"Ingested        : {state['ingested']:,}")
    logger.info(f"Skipped (noise) : {state['skipped']:,}")
    logger.info(f"  public        : {state['by_level']['public']:,}")
    logger.info(f"  internal      : {state['by_level']['internal']:,}")
    logger.info(f"  confidential  : {state['by_level']['confidential']:,}")
    logger.info(f"Time elapsed    : {elapsed/60:.1f} minutes")
    logger.info(f"Collection      : {COLLECTION_NAME} in AstraDB")
    logger.info("=" * 60)

    # Clean up checkpoint on successful completion
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("Checkpoint cleared.")


if __name__ == "__main__":
    ingest()
