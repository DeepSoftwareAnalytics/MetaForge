import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import pytest

from rho.agent.container_codex import ContainerCodexAgent
from rho.agent.judge_output import usable_judge_output
from rho.agent.round_recovery import RoundRecovery
from rho.protocols import Trajectory


def trajectory(tid, text, **kwargs):
    return Trajectory(id=tid, kind='evaluate', task_id='task', harness_id='a->b',
                      instructions='judge', events=[], final_message=text,
                      stdout='', stderr='', workspace_diff={},
                      workspace_deletions=frozenset(), exit_code=0,
                      wall_time_s=1, **kwargs)


@pytest.mark.parametrize('value', range(-10, 11))
def test_full_upstream_score_range(value):
    assert usable_judge_output('evaluate', json.dumps({'value': value, 'rationale': 'comparison'}), 0)


@pytest.mark.parametrize('text', ['', 'done', '{}', '[]', 'null', '{"value": 3}',
    '{"value": true, "rationale": "x"}', '{"value": 2.5, "rationale": "x"}',
    '{"value": "3", "rationale": "x"}', '{"value": 11, "rationale": "x"}',
    '{"value": -11, "rationale": "x"}', '{"value": 0, "rationale": " "}'])
def test_normal_exit_without_valid_result_is_not_success(text):
    assert not usable_judge_output('evaluate', text, 0)


def test_timeout_or_error_with_valid_json_is_not_success():
    text = '{"value": 3, "rationale": "comparison"}'
    assert not usable_judge_output('evaluate', text, 1)
    assert not usable_judge_output('evaluate', text, 0, True)


def recovery(tmp_path):
    handoff = tmp_path / 'handoff.json'
    handoff.write_text(json.dumps({'draining_jobs': [], 'previous_limit': 8}))
    (tmp_path / 'initial_solve_recovery.json').write_text('{"entries": []}')
    return RoundRecovery(tmp_path, handoff)


def test_judge_retries_do_not_block_queue_and_published_result_wins(tmp_path):
    r = recovery(tmp_path)
    empty = trajectory('empty', '')
    first = trajectory('first', '{"value": -3, "rationale": "first"}')
    later = trajectory('later', '{"value": 9, "rationale": "later"}')
    for tr in [empty, first, later]:
        r.checkpoints.put(tr)
    key = ('evaluate', 'task', 'a->b', None)
    assert r.checkpoint_map()[key] == 'first'  # not the most favorable score
    with r.slot():
        assert r.inflight == 1
    (tmp_path / 'retry_queue').mkdir()
    (tmp_path / 'retry_queue/accepted.json').write_text(json.dumps({
        'state': 'completed', 'trajectory_id': 'later'}))
    assert r.checkpoint_map()[key] == 'later'
    (tmp_path / 'retry_queue/accepted.json').unlink()
    r.published.put(later)
    assert r.checkpoint_map()[key] == 'later'
    ws = tmp_path / 'workspace'; ws.mkdir()
    assert r.take(ws, 'judge', kind='evaluate', task_id='task', harness_id='a->b').id == 'later'
    with pytest.raises(RuntimeError, match='instructions changed'):
        r.take(ws, 'changed', kind='evaluate', task_id='task', harness_id='a->b')


def test_invalid_checkpoint_cannot_be_replayed(tmp_path):
    r = recovery(tmp_path)
    r.checkpoints.put(trajectory('empty', ''))
    assert r.checkpoint_map() == {}
    with r.slot():
        pass


def test_duplicate_solve_still_requires_disambiguation(tmp_path):
    r = recovery(tmp_path)
    for tid in ['one', 'two']:
        r.checkpoints.put(replace(trajectory(tid, 'done'), kind='solve'))
    with pytest.raises(RuntimeError, match='Ambiguous'):
        r.checkpoint_map()


def test_agent_retries_empty_output_and_accepts_upstream_score(monkeypatch, tmp_path):
    import rho.agent.container_codex as module
    monkeypatch.setattr(module, 'global_codex_pool', lambda: type('Pool', (), {'acquire': lambda self: nullcontext()})())
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: None)
    agent = object.__new__(ContainerCodexAgent)
    calls = []
    outputs = iter(['', '{"value": 4, "rationale": "comparison"}'])
    def run(*args, **kwargs):
        calls.append(kwargs)
        return trajectory(str(len(calls)), next(outputs))
    monkeypatch.setattr(agent, '_run', run)
    result = agent.run(tmp_path, 'judge', kind='evaluate')
    assert json.loads(result.final_message)['value'] == 4 and len(calls) == 2
    monkeypatch.setattr(agent, '_run', lambda *a, **k: trajectory('empty', ''))
    with pytest.raises(RuntimeError, match='unusable after 3'):
        agent.run(tmp_path, 'judge', kind='evaluate')
