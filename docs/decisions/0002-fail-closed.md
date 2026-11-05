# 0002: Fail closed when a check errors or exceeds its latency budget

- Status: accepted
- Date: 2026-11-19

## Context

The engine runs several detectors on every request and response (docs/design/policy.md). Any of them can fail:
- a detector raises (a model fails to load, Presidio hits an edge case, a bug);
- a detector returns output the wrapper cannot interpret (an unknown reason string, a span outside the text);
- the checks together take longer than the policy's `latency_budget_ms`;
- the request contains content no detector can read (an image, an unknown role).

In each case the engine has no verdict for some of the text. It must still return allow, redact or block, and whatever it returns is a security decision.

v0 got this wrong: LLM errors were reported as "Passed" ([v0-findings.md](../v0-findings.md), finding 2). A firewall that passes traffic whenever it breaks can be bypassed by making it break: an oversized or unusual input, or simply load.

## Options

1. **Fail open:** on a failure, ignore the detector that failed and decide from the rest.
   - Pro: a detector outage never blocks users.
   - Con: an attacker who can trigger a failure or a slowdown gets unchecked traffic through. With `pii` failing, personal data is forwarded unredacted. Failures are silent unless someone watches the logs.
2. **Fail closed:** on a failure, block.
   - Pro: no input passes without the checks the policy asked for. Failures are visible at once, as blocked requests.
   - Con: a broken or slow detector blocks traffic until it is fixed. A latency budget set too low causes false blocks.
3. **Fail closed by default, fail open per policy.**
   - Pro: the safe behavior is the default. An operator who prefers availability for a particular group can say so explicitly, in a file that is reviewed.
   - Con: two behaviors to test and document.

## Decision

Option 3. `on_detector_error` and `on_timeout` both default to `fail_closed`, and both shipped policies (`default.yaml`, `strict.yaml`) set them to `fail_closed` explicitly. Content no detector can read is blocked even under `fail_open`, because allowing it would be a bypass ("put the payload in an image") rather than an outage.

Two further rules keep failures from ever loosening a decision:
- a failure never lowers the running action: under `fail_open` the failed detector contributes nothing, but earlier findings stand;
- when two policies are merged for a caller (user and team both mapped), the result fails closed if either policy does.

## Consequences

- A detector outage or a slow model shows up as blocked requests, not as silent gaps. The audit log (later task) records each failure's detector, kind and exception class, never the exception message, which may contain the input.
- The latency budget must be set from measurement. `1000 ms` in the shipped policies is an unmeasured placeholder until the latency task.
- The budget is checked between detector calls, not enforced as a hard deadline. One slow call runs to completion before the request fails closed, so a hung model holds the request rather than letting it through. The latency task will add a hard deadline.
- An operator who chooses `fail_open` for a policy accepts that, during a failure, text may pass without redaction. The design doc says so next to the setting.

## Evidence

- `tests/unit/test_pipeline.py`:
  - `test_detector_error_fails_closed_by_default` and `test_detector_error_fail_closed_skips_later_cost_classes`: an exception blocks.
  - `test_span_outside_the_text_is_invalid_output_and_fails_closed`: invalid output blocks.
  - `test_timeout_fails_closed_by_default`, `test_timeout_counts_time_across_all_detectors` and `test_timeout_stops_a_detector_between_messages`: exceeding the budget blocks.
  - `test_detector_error_fail_open_never_lowers_an_earlier_block` and `test_findings_of_the_call_that_overran_are_kept`: `fail_open` never lowers the decision.
- `tests/unit/test_engine.py::test_unsupported_input_is_blocked_even_when_failing_open`: unreadable content is blocked under `fail_open`.
- `tests/unit/test_policy.py::test_failure_modes_default_to_fail_closed` and `test_shipped_policy_is_valid_for_the_builtin_detectors`: the defaults and both shipped policies fail closed.
- `tests/unit/test_identity.py::test_user_mapping_can_never_weaken_the_team_policy`: a merged policy fails closed if either input does.
