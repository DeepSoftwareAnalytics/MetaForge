from pathlib import Path
import runpy


def test_final_runs_before_baseline_backlog_and_failures_do_not_stop_other_arm():
    module = runpy.run_path(str(Path(__file__).resolve().parents[2] /
                               'scripts/continue-clean-heldout.py'))
    completed, calls, events, sleeps = set(), [], [], []
    def execute(arm):
        calls.append(arm)
        if calls.count(arm) == 2:
            completed.add(arm)
            return 0
        return 1
    module['continue_arms'](lambda arm: arm in completed, execute,
        lambda phase, **kw: events.append((phase, kw)), sleep=sleeps.append)
    assert calls == ['final', 'baseline', 'final', 'baseline']
    assert sleeps == [60]
    assert events[-1][0] == 'complete'


def test_successful_arm_is_not_reexecuted():
    module = runpy.run_path(str(Path(__file__).resolve().parents[2] /
                               'scripts/continue-clean-heldout.py'))
    completed, calls = {'baseline'}, []
    def execute(arm):
        calls.append(arm)
        completed.add(arm)
        return 0
    module['continue_arms'](lambda arm: arm in completed, execute, lambda *a, **k: None)
    assert calls == ['final']
