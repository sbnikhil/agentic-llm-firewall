import json
import time
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.graph.nodes import input_security_node, output_security_node

def load_audit_logs():
    log_file = Path(__file__).parent.parent / "security_audit.log"
    logs = []
    if log_file.exists():
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    logs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    return logs

def calculate_security_metrics(logs):
    if not logs:
        return None
    
    total = len(logs)
    blocked = sum(1 for log in logs if log.get("blocked"))
    
    block_reasons = defaultdict(int)
    for log in logs:
        if log.get("blocked"):
            reason = log.get("block_reason", "unknown")
            block_reasons[reason] += 1
    
    topic_distribution = defaultdict(int)
    for log in logs:
        topic = log.get("topic_level", "unknown")
        topic_distribution[topic] += 1
    
    return {
        "total_interactions": total,
        "blocked_count": blocked,
        "allowed_count": total - blocked,
        "block_rate": round((blocked / total) * 100, 2),
        "block_reasons": dict(block_reasons),
        "topic_distribution": dict(topic_distribution)
    }

def load_golden_dataset():
    dataset_path = Path(__file__).parent.parent / "tests" / "data" / "golden_dataset.json"
    with open(dataset_path, 'r') as f:
        return json.load(f)

def evaluate_on_golden_dataset():
    dataset = load_golden_dataset()
    
    results = {
        "redaction": {"correct": 0, "total": 0, "false_positives": 0},
        "leak_detection": {"correct": 0, "total": 0, "false_negatives": 0, "false_positives": 0},
        "input_block": {"correct": 0, "total": 0, "false_negatives": 0, "false_positives": 0}
    }
    
    for case in dataset:
        if case["expected_type"] == "redaction":
            results["redaction"]["total"] += 1
            state = {"user_input": case["input"], "user_role": "employee", "loop_count": 0}
            result = input_security_node(state)
            
            if case["should_not_contain"] not in result.get("redacted_input", ""):
                results["redaction"]["correct"] += 1
        
        if case["expected_type"] == "input_block":
            results["input_block"]["total"] += 1
            state = {"user_input": case["input"], "user_role": "employee", "loop_count": 0}
            result = input_security_node(state)
            
            should_block = case.get("should_be_blocked", False)
            is_blocked = result.get("blocked", False)
            
            if is_blocked == should_block:
                results["input_block"]["correct"] += 1
            elif is_blocked and not should_block:
                results["input_block"]["false_positives"] += 1
            elif not is_blocked and should_block:
                results["input_block"]["false_negatives"] += 1
        
        if case["expected_type"] in ["audit_fail", "pass"]:
            results["leak_detection"]["total"] += 1
            state = {"llm_response": case["input"], "loop_count": 0}
            result = output_security_node(state)
            
            should_block = case.get("should_be_blocked", False)
            is_blocked = result.get("blocked", False) or not result.get("passed_output_security", True)
            
            if is_blocked == should_block:
                results["leak_detection"]["correct"] += 1
            elif is_blocked and not should_block:
                results["leak_detection"]["false_positives"] += 1
            elif not is_blocked and should_block:
                results["leak_detection"]["false_negatives"] += 1
    
    redaction_accuracy = (results["redaction"]["correct"] / results["redaction"]["total"] * 100) if results["redaction"]["total"] > 0 else 0
    leak_accuracy = (results["leak_detection"]["correct"] / results["leak_detection"]["total"] * 100) if results["leak_detection"]["total"] > 0 else 0
    input_accuracy = (results["input_block"]["correct"] / results["input_block"]["total"] * 100) if results["input_block"]["total"] > 0 else 0
    
    fpr_leak = (results["leak_detection"]["false_positives"] / results["leak_detection"]["total"] * 100) if results["leak_detection"]["total"] > 0 else 0
    fnr_leak = (results["leak_detection"]["false_negatives"] / results["leak_detection"]["total"] * 100) if results["leak_detection"]["total"] > 0 else 0
    
    fpr_input = (results["input_block"]["false_positives"] / results["input_block"]["total"] * 100) if results["input_block"]["total"] > 0 else 0
    fnr_input = (results["input_block"]["false_negatives"] / results["input_block"]["total"] * 100) if results["input_block"]["total"] > 0 else 0
    
    return {
        "redaction_accuracy": round(redaction_accuracy, 2),
        "leak_detection_accuracy": round(leak_accuracy, 2),
        "input_security_accuracy": round(input_accuracy, 2),
        "false_positive_rate": round(fpr_leak, 2),
        "false_negative_rate": round(fnr_leak, 2),
        "input_fpr": round(fpr_input, 2),
        "input_fnr": round(fnr_input, 2),
        "details": results
    }

def print_report():
    """Generate evaluation report with metrics from golden dataset"""
    print("=" * 60)
    print("AI SECURITY GATEWAY - EVALUATION REPORT")
    print("=" * 60)
    
    logs = load_audit_logs()
    if logs:
        print("\n[PRODUCTION METRICS]")
        security_metrics = calculate_security_metrics(logs)
        for key, value in security_metrics.items():
            print(f"{key}: {value}")
    else:
        print("\n[No production logs found]")
    
    print("\n[GOLDEN DATASET EVALUATION]")
    eval_results = evaluate_on_golden_dataset()
    print(f"PII Redaction Accuracy: {eval_results['redaction_accuracy']}%")
    print(f"Input Security Accuracy: {eval_results['input_security_accuracy']}%")
    print(f"Leak Detection Accuracy: {eval_results['leak_detection_accuracy']}%")
    print(f"Output False Positive Rate: {eval_results['false_positive_rate']}%")
    print(f"Output False Negative Rate: {eval_results['false_negative_rate']}%")
    print(f"Input False Positive Rate: {eval_results['input_fpr']}%")
    print(f"Input False Negative Rate: {eval_results['input_fnr']}%")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    print_report()
