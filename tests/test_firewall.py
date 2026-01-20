import pytest
import json
from app.graph.nodes import input_security_node, output_security_node

def load_dataset():
    with open("tests/data/golden_dataset.json", "r") as f:
        return json.load(f)

@pytest.mark.parametrize("case", load_dataset())
def test_firewall_logic(case):
    if case["expected_type"] == "redaction":
        initial_state = {
            "user_input": case["input"],
            "user_role": "employee",
            "loop_count": 0
        }
        result = input_security_node(initial_state)
        
        safe_text = result.get("redacted_input", "")
        
        assert safe_text != "", f"Expected 'redacted_input' key in {result}"
        assert case["should_not_contain"] not in safe_text
        assert "<" in safe_text or case["should_not_contain"] not in safe_text

    if case["expected_type"] in ["audit_fail", "pass"]:
        state = {
            "llm_response": case["input"],
            "loop_count": 0
        }
        result = output_security_node(state)
        
        should_block = case.get("should_be_blocked", False)
        is_blocked = result.get("blocked", False) or not result.get("passed_output_security", True)
        assert is_blocked == should_block