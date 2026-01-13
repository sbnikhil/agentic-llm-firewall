import pytest
import json
from app.graph.nodes import redactor_node, auditor_node
# BaseMessage import isn't strictly needed if we use the dict key directly, 
# but good to keep for robust code.

def load_dataset():
    with open("tests/data/golden_dataset.json", "r") as f:
        return json.load(f)

@pytest.mark.parametrize("case", load_dataset())
def test_firewall_logic(case):
    # 1. Test Redaction Node
    if case["expected_type"] == "redaction":
        initial_state = {
            "messages": [case["input"]], 
            "user_input": case["input"], # What the node takes in
            "loop_count": 0
        }
        result = redactor_node(initial_state)
        
        # [FIX] Use the key your node actually returns!
        safe_text = result.get("redacted_input", "")
        
        assert safe_text != "", f"Expected 'redacted_input' key in {result}"
        assert case["should_not_contain"] not in safe_text
        assert "<" in safe_text 

    # 2. Test Auditor Node (Already passing!)
    if case["expected_type"] in ["audit_fail", "pass"]:
        state = {
            "messages": [case["input"]], 
            "llm_response": case["input"], 
            "is_leak": False, 
            "loop_count": 0
        }
        result = auditor_node(state)
        assert result["is_leak"] == case.get("should_be_blocked", False)