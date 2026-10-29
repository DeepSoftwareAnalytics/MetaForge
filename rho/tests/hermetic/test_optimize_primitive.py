import json

from rho.agent.cache import AgentResponseCache, CachingAgent
from rho.agent.fake import FakeAgent, FakeResponse
from rho.orchestrators.diagnose import (
    DIAGNOSE_INSTRUCTIONS,
    DIAGNOSE_NO_CONSISTENCY_INSTRUCTIONS,
    DIAGNOSE_NO_VALIDATION_INSTRUCTIONS,
    _parse_diagnosis,
    diagnose,
)
from rho.orchestrators.solve import solve
from rho.protocols import Diagnosis, Trajectory, TrajectoryAnalysis
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore
from rho.strategies._common import optimize_agent_call
from rho.strategies.diagnose import DiagnoseStrategy, _dump_diagnosis
from tests.helpers import make_fake_agent


def _diagnose_trajectory(final_message: str, *, traj_id: str = "diag_traj") -> Trajectory:
    return Trajectory(
        id=traj_id,
        kind="diagnose",
        task_id="task",
        harness_id="harness",
        instructions="",
        events=[],
        final_message=final_message,
        stdout="",
        stderr="",
        workspace_diff={},
        workspace_deletions=frozenset(),
        exit_code=0,
        wall_time_s=0.0,
    )


def _diagnosis_payload(**overrides):
    payload = {
        "task_id": "ignored",
        "trajectory_analyses": [
            {
                "trajectory": "trajectory_0",
                "successful": 1,
                "quality_analysis": "Completed accurately and efficiently.",
                "issues": "",
            }
        ],
        "failure_mode_analysis": "",
        "inconsistency_analysis": "",
        "harness_improvement_direction": "",
    }
    payload.update(overrides)
    return payload


def test_no_consistency_diagnose_prompt_removes_consistency_signal() -> None:
    assert "Analyze inconsistency" in DIAGNOSE_INSTRUCTIONS
    assert "inconsistency_analysis" in DIAGNOSE_INSTRUCTIONS

    lowered = DIAGNOSE_NO_CONSISTENCY_INSTRUCTIONS.lower()
    assert "inconsistency" not in lowered
    assert "consistency" not in lowered
    assert "consistent" not in lowered
    assert "inconsistency_analysis" not in DIAGNOSE_NO_CONSISTENCY_INSTRUCTIONS
    assert "Compare the three event sequences" not in DIAGNOSE_NO_CONSISTENCY_INSTRUCTIONS


def test_no_validation_diagnose_prompt_keeps_inconsistency_step_verbatim() -> None:
    assert "Analyze inconsistency" in DIAGNOSE_NO_VALIDATION_INSTRUCTIONS
    assert "inconsistency_analysis" in DIAGNOSE_NO_VALIDATION_INSTRUCTIONS
    assert "trajectory_analyses" not in DIAGNOSE_NO_VALIDATION_INSTRUCTIONS
    assert "failure_mode_analysis" not in DIAGNOSE_NO_VALIDATION_INSTRUCTIONS

    full_body = _section_body(
        DIAGNOSE_INSTRUCTIONS,
        "## Step 3: Analyze inconsistency",
        "## Step 4: Summarize harness improvement direction",
    )
    no_validation_body = _section_body(
        DIAGNOSE_NO_VALIDATION_INSTRUCTIONS,
        "## Step 2: Analyze inconsistency",
        "## Step 3: Summarize harness improvement direction",
    )

    assert no_validation_body == full_body


def test_optimize_primitive_captures_new_harness(fake_agent_default_scripts, toy_dataset_root, tmp_path) -> None:
    from rho.datasets.directory import DirectoryTask

    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    harness = harness_store.empty()
    task = DirectoryTask(toy_dataset_root / "train" / "task_001", harness)
    trajs = [
        solve(fake_agent_default_scripts, task, harness, workdir=tmp_path / "workdir", sample_index=i)
        for i in range(3)
    ]
    result = DiagnoseStrategy().propose_candidates(
        agent=fake_agent_default_scripts,
        harness=harness,
        tasks_with_trajectories=[(task, trajs)],
        harness_store=harness_store,
        traj_store=FilesystemTrajectoryStore(tmp_path / "trajectories"),
        workdir=tmp_path / "workdir",
        n_samples=1,
        round_ix=0,
    )
    assert len(result.samples) == 1
    opt_traj = result.samples[0].optimize_trajectory
    new_harness = result.samples[0].candidate
    assert result.diagnoses is not None
    assert result.diagnose_trajectories is not None
    assert opt_traj.kind == "optimize"
    assert new_harness is not None
    assert new_harness.id != harness.id


def test_diagnose_uses_authoritative_task_id(toy_dataset_root, tmp_path) -> None:
    from rho.datasets.directory import DirectoryTask

    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    harness = harness_store.empty()
    task = DirectoryTask(toy_dataset_root / "train" / "task_001", harness)
    trajectories = [
        solve(
            FakeAgent({"solve": lambda workspace, instructions, output_schema: FakeResponse(final_message="team project code name is Phoenix")}),
            task,
            harness,
            workdir=tmp_path / "workdir",
            sample_index=i,
        )
        for i in range(3)
    ]
    agent = FakeAgent(
        {
            "diagnose": lambda workspace, instructions, output_schema: FakeResponse(
                final_message=json.dumps(
                    {
                        "task_id": "wrong_task_id",
                        "severity": 0.0,
                        "trajectory_analyses": [
                            {
                                "trajectory": "trajectory_0",
                                "successful": 1,
                                "quality_analysis": "Completed accurately and efficiently.",
                                "issues": "",
                            },
                            {
                                "trajectory": "trajectory_1",
                                "successful": 1,
                                "quality_analysis": "Completed accurately and efficiently.",
                                "issues": "",
                            },
                            {
                                "trajectory": "trajectory_2",
                                "successful": 1,
                                "quality_analysis": "Completed accurately and efficiently.",
                                "issues": "",
                            },
                        ],
                        "failure_mode_analysis": "",
                        "inconsistency_analysis": "",
                        "harness_improvement_direction": "",
                    }
                )
            )
        }
    )
    _diag_traj, diag = diagnose(agent, task, trajectories, harness, workdir=tmp_path / "workdir")
    assert diag.task_id == task.id
    assert diag.severity == 0.0


def test_diagnose_parses_valid_severity_float() -> None:
    tr = _diagnose_trajectory(json.dumps(_diagnosis_payload(severity=0.42)))

    diag = _parse_diagnosis("task_001", tr)

    assert diag.severity == 0.42


def test_diagnose_parses_numeric_string_severity() -> None:
    tr = _diagnose_trajectory(json.dumps(_diagnosis_payload(severity="0.3")))

    diag = _parse_diagnosis("task_001", tr)

    assert diag.severity == 0.3


def test_diagnose_clamps_severity() -> None:
    low = _parse_diagnosis(
        "task_001",
        _diagnose_trajectory(json.dumps(_diagnosis_payload(severity=-1))),
    )
    high = _parse_diagnosis(
        "task_001",
        _diagnose_trajectory(json.dumps(_diagnosis_payload(severity=2))),
    )

    assert low.severity == 0.0
    assert high.severity == 1.0


def test_diagnose_infers_legacy_missing_severity_from_content() -> None:
    clean = _parse_diagnosis(
        "task_001",
        _diagnose_trajectory(json.dumps(_diagnosis_payload())),
    )
    failing_payload = _diagnosis_payload(
        trajectory_analyses=[
            {
                "trajectory": "trajectory_0",
                "successful": 0,
                "quality_analysis": "Did not complete the task.",
                "issues": "Missing facts: project code name",
            }
        ],
        failure_mode_analysis="Missing facts: project code name",
    )
    failing = _parse_diagnosis(
        "task_001",
        _diagnose_trajectory(json.dumps(failing_payload)),
    )

    assert clean.severity == 0.0
    assert failing.severity == 1.0


def test_diagnose_invalid_json_fallback_has_high_severity() -> None:
    diag = _parse_diagnosis("task_001", _diagnose_trajectory("not json"))

    assert diag.severity == 1.0


def test_optimize_renders_and_sorts_diagnoses_by_severity(toy_dataset_root, tmp_path) -> None:
    from rho.datasets.directory import DirectoryTask

    captured_diagnoses: list[str] = []

    def optimize_script(workspace, instructions, output_schema):
        del instructions, output_schema
        for path in sorted((workspace / "diagnoses").iterdir()):
            captured_diagnoses.append((path / "diagnosis.md").read_text(encoding="utf-8"))
        return FakeResponse(final_message="no changes")

    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    harness = harness_store.empty()
    low_task = DirectoryTask(toy_dataset_root / "train" / "task_001", harness)
    high_task = DirectoryTask(toy_dataset_root / "train" / "task_002", harness)
    low = Diagnosis(
        task_id="low",
        trajectory_analyses=[],
        failure_mode_analysis="",
        inconsistency_analysis="",
        harness_improvement_direction="",
        severity=0.1,
    )
    high = Diagnosis(
        task_id="high",
        trajectory_analyses=[],
        failure_mode_analysis="Missing facts: oncall rotation",
        inconsistency_analysis="",
        harness_improvement_direction="Add facts to harness: oncall rotation",
        severity=0.9,
    )
    traj_store = FilesystemTrajectoryStore(tmp_path / "trajectories")

    def fake_diagnose(
        agent,
        task,
        trajectories,
        harness,
        *,
        workdir,
        stage=None,
        round_ix=None,
        include_consistency=True,
        include_validation=True,
    ):
        del agent, trajectories, harness, workdir, stage, round_ix
        del include_consistency, include_validation
        if task.id == low_task.id:
            return _diagnose_trajectory(
                json.dumps(_diagnosis_payload(severity=0.1)),
                traj_id="diag_low",
            ), low
        return _diagnose_trajectory(
            json.dumps(_diagnosis_payload(severity=0.9)),
            traj_id="diag_high",
        ), high

    import rho.strategies.diagnose as diagnose_strategy_module

    original_diagnose = diagnose_strategy_module.diagnose
    diagnose_strategy_module.diagnose = fake_diagnose
    try:
        result = DiagnoseStrategy().propose_candidates(
            agent=FakeAgent({"optimize": optimize_script}),
            harness=harness,
            tasks_with_trajectories=[
                (low_task, [_diagnose_trajectory("a"), _diagnose_trajectory("b"), _diagnose_trajectory("c")]),
                (high_task, [_diagnose_trajectory("a"), _diagnose_trajectory("b"), _diagnose_trajectory("c")]),
            ],
            harness_store=harness_store,
            traj_store=traj_store,
            workdir=tmp_path / "workdir",
            n_samples=1,
            round_ix=0,
        )
    finally:
        diagnose_strategy_module.diagnose = original_diagnose

    candidate = result.samples[0].candidate

    assert candidate is None
    assert len(captured_diagnoses) == 2
    assert "# Diagnosis: high" in captured_diagnoses[0]
    assert "**Severity:** 0.90" in captured_diagnoses[0]
    assert "# Diagnosis: low" in captured_diagnoses[1]
    assert "**Severity:** 0.10" in captured_diagnoses[1]


def test_diagnose_strategy_without_consistency_sanitizes_optimize_signal(
    monkeypatch,
    toy_dataset_root,
    tmp_path,
) -> None:
    from rho.datasets.directory import DirectoryTask

    captured: dict[str, object] = {}
    include_flags: list[bool] = []

    def optimize_script(workspace, instructions, output_schema):
        del output_schema
        captured["instructions"] = instructions
        captured["diagnosis_md"] = (
            workspace / "diagnoses" / "task_0000" / "diagnosis.md"
        ).read_text(encoding="utf-8")
        return FakeResponse(final_message="no changes")

    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    harness = harness_store.empty()
    task = DirectoryTask(toy_dataset_root / "train" / "task_001", harness)
    diagnosis = Diagnosis(
        task_id=task.id,
        trajectory_analyses=[],
        failure_mode_analysis="Missing facts: project code name",
        inconsistency_analysis="Two trajectories diverged on the answer source.",
        harness_improvement_direction="Add facts to harness: project code name",
        severity=0.9,
    )

    def fake_diagnose(
        agent,
        task,
        trajectories,
        harness,
        *,
        workdir,
        stage=None,
        round_ix=None,
        include_consistency=True,
        include_validation=True,
    ):
        del agent, task, trajectories, harness, workdir, stage, round_ix
        include_flags.append(include_consistency)
        assert include_validation is True
        return _diagnose_trajectory(
            json.dumps(_diagnosis_payload(severity=0.9)),
            traj_id="diag_no_consistency",
        ), diagnosis

    import rho.strategies.diagnose as diagnose_strategy_module

    monkeypatch.setattr(diagnose_strategy_module, "diagnose", fake_diagnose)

    result = DiagnoseStrategy(include_consistency=False).propose_candidates(
        agent=FakeAgent({"optimize": optimize_script}),
        harness=harness,
        tasks_with_trajectories=[
            (task, [_diagnose_trajectory("a"), _diagnose_trajectory("b"), _diagnose_trajectory("c")])
        ],
        harness_store=harness_store,
        traj_store=FilesystemTrajectoryStore(tmp_path / "trajectories"),
        workdir=tmp_path / "workdir",
        n_samples=1,
        round_ix=0,
    )

    assert include_flags == [False]
    assert result.diagnoses is not None
    assert result.diagnoses[0].inconsistency_analysis == ""
    assert result.samples[0].candidate is None
    assert "inconsistency" not in str(captured["instructions"]).lower()
    assert "consistency" not in str(captured["instructions"]).lower()
    assert "Inconsistency analysis" not in str(captured["diagnosis_md"])
    assert "Two trajectories diverged" not in str(captured["diagnosis_md"])


def test_diagnose_strategy_without_validation_sanitizes_optimize_signal(
    monkeypatch,
    toy_dataset_root,
    tmp_path,
) -> None:
    from rho.datasets.directory import DirectoryTask

    captured: dict[str, object] = {}
    include_flags: list[bool] = []

    def optimize_script(workspace, instructions, output_schema):
        del output_schema
        captured["instructions"] = instructions
        captured["diagnosis_md"] = (
            workspace / "diagnoses" / "task_0000" / "diagnosis.md"
        ).read_text(encoding="utf-8")
        return FakeResponse(final_message="no changes")

    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    harness = harness_store.empty()
    task = DirectoryTask(toy_dataset_root / "train" / "task_001", harness)
    diagnosis = Diagnosis(
        task_id=task.id,
        trajectory_analyses=[
            TrajectoryAnalysis(
                trajectory="trajectory_0",
                successful=0,
                quality_analysis="Did not answer the prompt completely.",
                issues="Missing facts: project code name",
            )
        ],
        failure_mode_analysis="Missing facts: project code name",
        inconsistency_analysis="Two trajectories diverged on the answer source.",
        harness_improvement_direction="Add facts to harness: project code name",
        severity=0.9,
    )

    def fake_diagnose(
        agent,
        task,
        trajectories,
        harness,
        *,
        workdir,
        stage=None,
        round_ix=None,
        include_consistency=True,
        include_validation=True,
    ):
        del agent, task, trajectories, harness, workdir, stage, round_ix
        assert include_consistency is True
        include_flags.append(include_validation)
        return _diagnose_trajectory(
            json.dumps(_diagnosis_payload(severity=0.9)),
            traj_id="diag_no_validation",
        ), diagnosis

    import rho.strategies.diagnose as diagnose_strategy_module

    monkeypatch.setattr(diagnose_strategy_module, "diagnose", fake_diagnose)

    result = DiagnoseStrategy(include_validation=False).propose_candidates(
        agent=FakeAgent({"optimize": optimize_script}),
        harness=harness,
        tasks_with_trajectories=[
            (task, [_diagnose_trajectory("a"), _diagnose_trajectory("b"), _diagnose_trajectory("c")])
        ],
        harness_store=harness_store,
        traj_store=FilesystemTrajectoryStore(tmp_path / "trajectories"),
        workdir=tmp_path / "workdir",
        n_samples=1,
        round_ix=0,
    )

    assert include_flags == [False]
    assert result.diagnoses is not None
    assert result.diagnoses[0].trajectory_analyses == []
    assert result.diagnoses[0].failure_mode_analysis == ""
    assert result.samples[0].candidate is None
    assert "inconsistency" in str(captured["instructions"]).lower()
    assert "failure mode" not in str(captured["instructions"]).lower()
    assert "per-trajectory" not in str(captured["instructions"]).lower()
    assert "Per-trajectory analysis" not in str(captured["diagnosis_md"])
    assert "Failure mode analysis" not in str(captured["diagnosis_md"])
    assert "Did not answer the prompt" not in str(captured["diagnosis_md"])
    assert "Two trajectories diverged" in str(captured["diagnosis_md"])


def test_optimize_with_cache_distinguishes_sample_index(toy_dataset_root, tmp_path) -> None:
    from rho.datasets.directory import DirectoryTask

    fake = make_fake_agent("good")
    agent = CachingAgent(fake, AgentResponseCache(tmp_path / "cache"))
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    harness = harness_store.empty()
    task = DirectoryTask(toy_dataset_root / "train" / "task_001", harness)
    diagnosis = Diagnosis(
        task_id=task.id,
        trajectory_analyses=[
            TrajectoryAnalysis(
                trajectory="trajectory_0",
                successful=0,
                quality_analysis="Missing facts: project code name",
                issues="Missing facts: project code name",
            )
        ],
        failure_mode_analysis="Missing facts: project code name",
        inconsistency_analysis="",
        harness_improvement_direction="Add facts to harness: project code name",
    )

    def build_workspace(ws):
        diagnoses_dir = ws / "diagnoses"
        diagnoses_dir.mkdir()
        _dump_diagnosis(diagnoses_dir / "task_0000", task, diagnosis)

    sample0_traj, sample0 = optimize_agent_call(
        agent,
        harness,
        harness_store,
        workspace_builder=build_workspace,
        instructions="Optimize harness.",
        workdir=tmp_path / "workdir",
        stage="round_optimize",
        round_ix=0,
        sample_index=0,
    )
    sample1_traj, sample1 = optimize_agent_call(
        agent,
        harness,
        harness_store,
        workspace_builder=build_workspace,
        instructions="Optimize harness.",
        workdir=tmp_path / "workdir",
        stage="round_optimize",
        round_ix=0,
        sample_index=1,
    )
    repeat_traj, repeat_sample1 = optimize_agent_call(
        agent,
        harness,
        harness_store,
        workspace_builder=build_workspace,
        instructions="Optimize harness.",
        workdir=tmp_path / "workdir",
        stage="round_optimize",
        round_ix=0,
        sample_index=1,
    )

    assert sample0 is not None
    assert sample1 is not None
    assert repeat_sample1 is not None
    assert sample0.id == sample1.id
    assert repeat_sample1.id == sample1.id
    assert sample0_traj.sample_index == 0
    assert sample1_traj.sample_index == 1
    assert repeat_traj.sample_index == 1
    assert len(fake.calls) == 2
    assert agent.hit_count == 1
    assert agent.miss_count == 2


def _section_body(text: str, header: str, next_header: str) -> str:
    start = text.index(header) + len(header)
    end = text.index(next_header, start)
    return text[start:end]
