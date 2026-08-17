"""
Injection Detection Evaluation
================================
Evaluates both layers of our injection defense against real, independently-sourced
attack datasets — not self-designed tests.

Two threat surfaces evaluated:

1. DIRECT INJECTION (user query)
   Dataset : deepset/prompt-injections (test split, 116 examples)
   What    : User types an injection attack in the query field
   Defender: validate_input()

2. INDIRECT INJECTION (retrieved document)
   Dataset : microsoft/llmail-inject-challenge (Phase1, 300 sampled)
   What    : Malicious instructions embedded inside a retrieved email/document
   Defender: scan_retrieved_document()

Benign control for FPR uses deepset label=0 examples (legitimate queries).

Usage:
    python scripts/evaluate_injection_detection.py
"""

import sys
import json
import random
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from app.security.input_validator import (
    detect_jailbreak_patterns,
    detect_csv_injection,
    contains_excessive_special_chars,
    detect_jailbreak_ml,
    validate_input,
    scan_retrieved_document,
)

LLMAIL_SAMPLE_SIZE = 300
RANDOM_SEED = 42


def load_direct_injection_data():
    """deepset/prompt-injections test split — labeled injection vs legitimate."""
    print("Loading deepset/prompt-injections (test split)...")
    ds = load_dataset("deepset/prompt-injections", split="test")
    injections = [row["text"] for row in ds if row["label"] == 1]
    legitimate = [row["text"] for row in ds if row["label"] == 0]
    print(f"  Injections : {len(injections)}")
    print(f"  Legitimate : {len(legitimate)}")
    return injections, legitimate


def load_indirect_injection_data():
    """LLMail-Inject Phase1 — real email-format indirect injection attacks."""
    print(f"Loading microsoft/llmail-inject-challenge (Phase1, sampling {LLMAIL_SAMPLE_SIZE})...")
    ds = load_dataset("microsoft/llmail-inject-challenge", split="Phase1")

    random.seed(RANDOM_SEED)
    indices = random.sample(range(len(ds)), LLMAIL_SAMPLE_SIZE)
    attacks = [ds[i]["body"] for i in indices if ds[i]["body"]]

    print(f"  Sampled attack emails : {len(attacks)}")
    return attacks


def blocked_by_patterns_only(text: str) -> bool:
    if detect_csv_injection(text):
        return True
    if detect_jailbreak_patterns(text):
        return True
    if contains_excessive_special_chars(text):
        return True
    return False


def evaluate_direct_injection(injections, legitimate):
    print("\nEvaluating direct injection detection (validate_input)...")
    print("  ML model loads on first call — takes ~10s...")

    results = {
        "pattern_only": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "pattern_plus_ml": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
    }
    missed = []

    for text in injections:
        if blocked_by_patterns_only(text):
            results["pattern_only"]["tp"] += 1
        else:
            results["pattern_only"]["fn"] += 1

        is_valid, _ = validate_input(text, user_role="employee")
        if not is_valid:
            results["pattern_plus_ml"]["tp"] += 1
        else:
            results["pattern_plus_ml"]["fn"] += 1
            missed.append(text[:100])

    for text in legitimate:
        if blocked_by_patterns_only(text):
            results["pattern_only"]["fp"] += 1
        else:
            results["pattern_only"]["tn"] += 1

        is_valid, _ = validate_input(text, user_role="employee")
        if not is_valid:
            results["pattern_plus_ml"]["fp"] += 1
        else:
            results["pattern_plus_ml"]["tn"] += 1

    return results, missed


def evaluate_indirect_injection(attacks, legitimate):
    print("\nEvaluating indirect injection detection (scan_retrieved_document)...")

    results = {
        "pattern_only": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "pattern_plus_ml": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
    }
    missed = []

    for text in attacks:
        # Pattern-only layer
        if detect_jailbreak_patterns(text):
            results["pattern_only"]["tp"] += 1
        else:
            results["pattern_only"]["fn"] += 1

        # Full scanner
        is_suspicious, _ = scan_retrieved_document(text)
        if is_suspicious:
            results["pattern_plus_ml"]["tp"] += 1
        else:
            results["pattern_plus_ml"]["fn"] += 1
            missed.append(text[:100])

    for text in legitimate:
        if detect_jailbreak_patterns(text):
            results["pattern_only"]["fp"] += 1
        else:
            results["pattern_only"]["tn"] += 1

        is_suspicious, _ = scan_retrieved_document(text)
        if is_suspicious:
            results["pattern_plus_ml"]["fp"] += 1
        else:
            results["pattern_plus_ml"]["tn"] += 1

    return results, missed


def print_section(title, dataset_source, results, missed):
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print(f"  Dataset: {dataset_source}")
    print(f"{'=' * 62}")

    for layer, label in [
        ("pattern_only",   "Layer 1 — Pattern matching only    "),
        ("pattern_plus_ml","Layer 2 — Pattern + ML (DeBERTa)   "),
    ]:
        r = results[layer]
        total_attacks = r["tp"] + r["fn"]
        total_benign  = r["fp"] + r["tn"]
        dr  = r["tp"] / total_attacks * 100 if total_attacks > 0 else 0
        fpr = r["fp"] / total_benign  * 100 if total_benign  > 0 else 0
        print(f"\n  {label}")
        print(f"    Detection rate : {r['tp']}/{total_attacks} = {dr:.1f}%")
        print(f"    False pos rate : {r['fp']}/{total_benign}  = {fpr:.1f}%")

    p = results["pattern_only"]
    m = results["pattern_plus_ml"]
    total_attacks = p["tp"] + p["fn"]
    total_benign  = p["fp"] + p["tn"]
    dr_gain  = (m["tp"] - p["tp"]) / total_attacks * 100 if total_attacks > 0 else 0
    fpr_delta = (m["fp"] - p["fp"]) / total_benign  * 100 if total_benign  > 0 else 0

    print(f"\n  ML layer gain  : detection +{dr_gain:.1f}%  |  FPR {fpr_delta:+.1f}%")

    if missed:
        print(f"\n  Missed by both layers ({len(missed)} examples):")
        for ex in missed[:3]:
            print(f"    \"{ex}\"")


def main():
    print("=" * 62)
    print("  INJECTION DETECTION — FULL EVALUATION")
    print("=" * 62)

    # Load data
    direct_attacks, legitimate = load_direct_injection_data()
    indirect_attacks = load_indirect_injection_data()
    print()

    # Evaluate
    direct_results, direct_missed = evaluate_direct_injection(direct_attacks, legitimate)
    indirect_results, indirect_missed = evaluate_indirect_injection(indirect_attacks, legitimate)

    # Report
    print_section(
        "DIRECT INJECTION (user query)",
        "deepset/prompt-injections — test split",
        direct_results,
        direct_missed,
    )
    print_section(
        "INDIRECT INJECTION (retrieved document)",
        "microsoft/llmail-inject-challenge — Phase1 sample",
        indirect_results,
        indirect_missed,
    )

    print(f"\n{'=' * 62}\n")

    # Save
    output_path = Path(__file__).parent.parent / "evaluation_results_injection.json"
    with open(output_path, "w") as f:
        json.dump({
            "direct_injection": direct_results,
            "indirect_injection": indirect_results,
        }, f, indent=2)
    print(f"  Results saved to: {output_path}\n")


if __name__ == "__main__":
    main()
