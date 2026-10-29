"""Validate completed judge output before accepting or replaying a call."""
import json

from rho.orchestrators._util import extract_json_object


def usable_judge_output(kind, final_message, exit_code, timed_out=False):
    if exit_code != 0 or timed_out or not final_message.strip():
        return False
    try:
        result = json.loads(extract_json_object(final_message))
    except (ValueError, TypeError):
        return False
    if not isinstance(result, dict):
        return False
    if kind == 'evaluate':
        # Match upstream EVAL_INSTRUCTIONS; never coerce a boolean or fraction
        # into an integer score, or turn a missing result into a tie.
        value = result.get('value')
        return (type(value) is int and -10 <= value <= 10
                and isinstance(result.get('rationale'), str)
                and bool(result['rationale'].strip()))
    if kind == 'diagnose':
        return 'trajectory_analyses' in result
    raise ValueError('Not a judge call: ' + kind)
