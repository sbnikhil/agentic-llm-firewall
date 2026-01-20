import json
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List

METRICS_FILE = Path(__file__).parent.parent / "metrics.json"

class MetricsCollector:
    def __init__(self):
        self.metrics = defaultdict(list)
        
    def record_request(self, user_id: str, user_role: str, topic_level: str, 
                      blocked: bool, block_reason: str, latency_ms: float,
                      leak_score: float, loop_count: int):
        event = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "user_role": user_role,
            "topic_level": topic_level,
            "blocked": blocked,
            "block_reason": block_reason,
            "latency_ms": latency_ms,
            "leak_score": leak_score,
            "loop_count": loop_count
        }
        self.metrics["requests"].append(event)
        self._save_metrics()
    
    def _save_metrics(self):
        with open(METRICS_FILE, 'a') as f:
            f.write(json.dumps(self.metrics) + "\n")
    
    def get_summary_stats(self) -> Dict:
        if not self.metrics["requests"]:
            return {}
        
        requests = self.metrics["requests"]
        total = len(requests)
        blocked = sum(1 for r in requests if r["blocked"])
        
        latencies = [r["latency_ms"] for r in requests]
        leak_scores = [r["leak_score"] for r in requests if r["leak_score"] > 0]
        
        return {
            "total_requests": total,
            "blocked_requests": blocked,
            "block_rate": blocked / total if total > 0 else 0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            "p99_latency_ms": sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0,
            "avg_leak_score": sum(leak_scores) / len(leak_scores) if leak_scores else 0,
            "max_leak_score": max(leak_scores) if leak_scores else 0
        }

metrics_collector = MetricsCollector()
