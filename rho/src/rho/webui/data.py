from __future__ import annotations

import difflib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rho.webui.trace import build_trace_payload, load_events, read_command_output_chunk

_MISSING = object()


class RunRepository:
    def __init__(self, runs_root: Path) -> None:
        self.runs_root = runs_root.resolve()

    def list_runs(self) -> list[dict[str, Any]]:
        runs = [self._summarize_run(path) for path in self._iter_run_dirs()]
        runs.sort(key=lambda run: run["name"], reverse=True)
        return runs

    def get_run(self, run_id: str) -> dict[str, Any]:
        run_dir = self._resolve_run(run_id)
        manifest = self._read_manifest(run_dir)
        usage = self._read_usage_summary(run_dir)
        summary = self._read_json(run_dir / "reports" / "summary.json")
        config = self._read_json(run_dir / "config.json")
        environment = self._read_json(run_dir / "environment.json")
        trajectory_summaries = [
            self._read_trajectory_summary(run_dir, meta_path.parent.name)
            for meta_path in sorted((run_dir / "trajectories").glob("*/meta.json"))
        ]
        round_summaries = [
            self._read_round_summary(run_dir, round_dir)
            for round_dir in sorted((run_dir / "rounds").glob("round_*"))
            if round_dir.is_dir()
        ]
        selection = self._read_json(run_dir / "selection.json", default=None)
        selection_summary = None
        if isinstance(selection, dict):
            selected_ids = selection.get("selected_task_ids") if isinstance(selection.get("selected_task_ids"), list) else []
            all_candidate_ids = selection.get("all_candidate_ids") if isinstance(selection.get("all_candidate_ids"), list) else []
            selection_summary = {
                "selector": selection.get("selector"),
                "k": selection.get("k", len(selected_ids)),
                "candidate_count": len(all_candidate_ids),
            }
        return {
            "id": run_id,
            "name": run_dir.name,
            "path": str(run_dir),
            "config": config,
            "environment": environment,
            "summary": summary,
            "manifest": manifest,
            "usage_summary": usage,
            "reports": self._read_report_files(run_dir),
            "rounds": round_summaries,
            "trajectories": trajectory_summaries,
            "selection_present": selection_summary is not None,
            "selection_summary": selection_summary,
        }

    def get_round(self, run_id: str, round_ix: int) -> dict[str, Any]:
        run_dir = self._resolve_run(run_id)
        round_dir = run_dir / "rounds" / f"round_{round_ix}"
        if not round_dir.exists():
            raise KeyError(f"Unknown round {round_ix}")
        config = self._read_json(run_dir / "config.json", default={})
        strategy = config.get("optimize_strategy", "diagnosis")
        solve_before_groups = self._read_json(round_dir / "solve_before_traj_ids.json", default=[])
        solve_before = [
            [self._read_trajectory_summary(run_dir, traj_id) for traj_id in group]
            for group in solve_before_groups
            if isinstance(group, list)
        ]
        optimize_candidates = self._read_json(round_dir / "optimize_candidates.json", default={})
        diagnose_ids = self._read_json(round_dir / "diagnose_traj_ids.json", default=[])
        solve_after_ids = self._read_json(round_dir / "solve_after_traj_ids.json", default=[])
        eval_ids = self._read_json(round_dir / "eval_traj_ids.json", default=[])
        optimize_ids = _optimize_traj_ids(round_dir, optimize_candidates)
        optimize_summaries = [
            self._read_trajectory_summary(run_dir, traj_id)
            for traj_id in optimize_ids
            if isinstance(traj_id, str) and not traj_id.startswith("(")
        ]
        optimize_summary = optimize_summaries[0] if optimize_summaries else None
        return {
            "round_ix": round_ix,
            "strategy": strategy,
            "input_harness_id": self._read_text(round_dir / "input_harness_id"),
            "candidate_harness_id": self._read_text(round_dir / "candidate_harness_id"),
            "accepted": _parse_bool_text(self._read_text(round_dir / "accepted")),
            "mean_score": _parse_float_text(self._read_text(round_dir / "mean_score")),
            "optimize_candidates": optimize_candidates,
            "diagnoses": self._read_json(round_dir / "diagnoses.json", default=[]),
            "scores": self._read_json(round_dir / "scores.json", default=[]),
            "solve_before": solve_before,
            "diagnose_trajectories": [
                self._read_trajectory_summary(run_dir, traj_id) for traj_id in diagnose_ids if isinstance(traj_id, str)
            ],
            "optimize_trajectory": optimize_summary,
            "optimize_trajectories": optimize_summaries,
            "solve_after_trajectories": [
                self._read_trajectory_summary(run_dir, traj_id) for traj_id in solve_after_ids if isinstance(traj_id, str)
            ],
            "evaluate_trajectories": [
                self._read_trajectory_summary(run_dir, traj_id) for traj_id in eval_ids if isinstance(traj_id, str)
            ],
            "candidate_harness_diff": self._read_candidate_harness_diff(run_dir, round_dir),
        }

    def get_trajectory(self, run_id: str, traj_id: str) -> dict[str, Any]:
        run_dir = self._resolve_run(run_id)
        traj_dir = self._resolve_trajectory(run_dir, traj_id)
        meta = self._read_json(traj_dir / "meta.json", default={})
        workspace_files = []
        diff_dir = traj_dir / "workspace_diff"
        if diff_dir.exists():
            for path in sorted(diff_dir.rglob("*")):
                if path.is_file():
                    workspace_files.append(
                        {
                            "path": path.relative_to(diff_dir).as_posix(),
                            "size": path.stat().st_size,
                        }
                    )
        return {
            "id": traj_id,
            "run_id": run_id,
            "meta": meta,
            "score": self.get_trajectory_score(run_id, traj_id),
            "instructions_excerpt": _read_text_excerpt(traj_dir / "instructions.md", max_chars=4096),
            "instructions_size": _file_size(traj_dir / "instructions.md"),
            "final_message_size": _file_size(traj_dir / "final_message.txt"),
            "stdout_size": _file_size(traj_dir / "stdout.log"),
            "stderr_size": _file_size(traj_dir / "stderr.log"),
            "events_size": _file_size(traj_dir / "events.jsonl"),
            "workspace_diff_files": workspace_files,
        }

    def get_trajectory_score(self, run_id: str, traj_id: str) -> dict[str, Any]:
        run_dir = self._resolve_run(run_id)
        traj_dir = self._resolve_trajectory(run_dir, traj_id)
        meta = self._read_json(traj_dir / "meta.json", default={})

        if meta.get("kind") == "evaluate":
            consistency_score = _read_consistency_score(traj_dir / "final_message.txt")
            if consistency_score is not None:
                return consistency_score

        for path in sorted((run_dir / "reports").glob("*grades*.json")):
            grades = self._read_json(path, default=[])
            if not isinstance(grades, list):
                continue
            score = _find_ctrf_score(grades, traj_id, source=f"reports/{path.name}")
            if score is not None:
                return score

        log_path = run_dir / "run.log"
        if log_path.exists():
            grades = _read_grades_from_run_log(log_path)
            score = _find_ctrf_score(grades, traj_id, source="run.log")
            if score is not None:
                return score

        return _ungraded_score()

    def get_run_tasks(self, run_id: str) -> list[dict[str, Any]]:
        run_dir = self._resolve_run(run_id)
        tasks: dict[str, dict[str, Any]] = {}
        for meta_path in sorted((run_dir / "trajectories").glob("*/meta.json")):
            meta = self._read_json(meta_path, default={})
            task_id = meta.get("task_id")
            if not isinstance(task_id, str) or not task_id or task_id == "*":
                continue
            entry = tasks.setdefault(task_id, {"task_id": task_id, "trajectory_count": 0, "stages": set()})
            entry["trajectory_count"] += 1
            stage = meta.get("stage")
            if isinstance(stage, str) and stage:
                entry["stages"].add(stage)
        return [
            {
                "task_id": task_id,
                "trajectory_count": entry["trajectory_count"],
                "stages": _sort_stages(entry["stages"]),
            }
            for task_id, entry in sorted(tasks.items())
        ]

    def get_task(self, run_id: str, task_id: str) -> dict[str, Any]:
        run_dir = self._resolve_run(run_id)
        trajectories = []
        for meta_path in sorted((run_dir / "trajectories").glob("*/meta.json")):
            meta = self._read_json(meta_path, default={})
            if meta.get("task_id") != task_id:
                continue
            summary = self._read_trajectory_summary(run_dir, meta_path.parent.name)
            if summary is None:
                continue
            summary["score"] = self.get_trajectory_score(run_id, meta_path.parent.name)
            trajectories.append(summary)
        if not trajectories:
            raise KeyError(task_id)
        trajectories.sort(key=_trajectory_sort_key)
        return {
            "task_id": task_id,
            "trajectories": trajectories,
            "diagnosis": self._find_task_diagnosis(run_dir, task_id),
            "selection": self._task_selection(run_dir, task_id),
        }

    def get_selection(self, run_id: str) -> dict[str, Any]:
        run_dir = self._resolve_run(run_id)
        selection = self._read_json(run_dir / "selection.json", default=None)
        if not isinstance(selection, dict):
            raise KeyError("selection")
        all_candidate_ids = selection.get("all_candidate_ids")
        if not isinstance(all_candidate_ids, list):
            all_candidate_ids = []
        selected_task_ids = selection.get("selected_task_ids")
        if not isinstance(selected_task_ids, list):
            selected_task_ids = []
        selected = {task_id for task_id in selected_task_ids if isinstance(task_id, str)}
        difficulty_scores = selection.get("difficulty_scores") if isinstance(selection.get("difficulty_scores"), dict) else {}
        fingerprints = selection.get("fingerprints") if isinstance(selection.get("fingerprints"), dict) else {}
        dpp_trace = self._read_json(run_dir / "selector_calls" / "dpp_trace.json", default=[])
        dpp_picks = {}
        if isinstance(dpp_trace, list):
            for pick in dpp_trace:
                if isinstance(pick, dict) and isinstance(pick.get("picked_id"), str):
                    dpp_picks[pick["picked_id"]] = {
                        "step": pick.get("step"),
                        "log_det_gain": pick.get("log_det_gain"),
                        "score": pick.get("score"),
                    }
        candidates = []
        for candidate_id in all_candidate_ids:
            if not isinstance(candidate_id, str):
                continue
            is_selected = candidate_id in selected
            candidates.append(
                {
                    "task_id": candidate_id,
                    "difficulty_score": difficulty_scores.get(candidate_id),
                    "fingerprint": fingerprints.get(candidate_id),
                    "selected": is_selected,
                    "dpp_pick": dpp_picks.get(candidate_id) if is_selected else None,
                }
            )
        return {
            "selector": selection.get("selector"),
            "k": selection.get("k"),
            "seed": selection.get("seed"),
            "theta": selection.get("theta"),
            "candidates": candidates,
            "selected_task_ids": selected_task_ids,
        }

    def get_harness(self, run_id: str, harness_id: str) -> dict[str, Any]:
        harness_dir = self._resolve_harness_dir(self._resolve_run(run_id), harness_id)
        files = [
            {"path": path.relative_to(harness_dir).as_posix(), "size": path.stat().st_size}
            for path in sorted(harness_dir.rglob("*"))
            if path.is_file()
        ]
        return {"harness_id": harness_id, "files": files}

    def get_harness_file_text(self, run_id: str, harness_id: str, rel_path: str) -> str:
        harness_dir = self._resolve_harness_dir(self._resolve_run(run_id), harness_id)
        candidate = (harness_dir / rel_path).resolve()
        if not candidate.is_file() or (harness_dir not in candidate.parents and candidate != harness_dir):
            raise KeyError(rel_path)
        return candidate.read_text(encoding="utf-8", errors="replace")

    def get_trace(self, run_id: str, traj_id: str) -> dict[str, Any]:
        run_dir = self._resolve_run(run_id)
        traj_dir = self._resolve_trajectory(run_dir, traj_id)
        events = load_events(traj_dir / "events.jsonl")
        return build_trace_payload(events, run_dir=run_dir)

    def get_trace_output_chunk(
        self,
        run_id: str,
        traj_id: str,
        step_id: str,
        *,
        start_line: int,
        max_lines: int,
    ) -> dict[str, Any]:
        run_dir = self._resolve_run(run_id)
        traj_dir = self._resolve_trajectory(run_dir, traj_id)
        events = load_events(traj_dir / "events.jsonl")
        return read_command_output_chunk(
            events,
            run_dir=run_dir,
            step_id=step_id,
            start_line=start_line,
            max_lines=max_lines,
        )

    def get_artifact_text(self, run_id: str, traj_id: str, artifact_name: str) -> str:
        run_dir = self._resolve_run(run_id)
        traj_dir = self._resolve_trajectory(run_dir, traj_id)
        artifact_map = {
            "instructions": traj_dir / "instructions.md",
            "final_message": traj_dir / "final_message.txt",
            "stdout": traj_dir / "stdout.log",
            "stderr": traj_dir / "stderr.log",
            "events_raw": traj_dir / "events.jsonl",
        }
        try:
            path = artifact_map[artifact_name]
        except KeyError as exc:
            raise KeyError(artifact_name) from exc
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def get_workspace_diff_text(self, run_id: str, traj_id: str, rel_path: str) -> str:
        run_dir = self._resolve_run(run_id)
        traj_dir = self._resolve_trajectory(run_dir, traj_id)
        base = (traj_dir / "workspace_diff").resolve()
        candidate = (base / rel_path).resolve()
        if not candidate.is_file() or (base not in candidate.parents and candidate != base):
            raise KeyError(rel_path)
        return candidate.read_text(encoding="utf-8", errors="replace")

    def get_report_text(self, run_id: str, report_name: str) -> str:
        run_dir = self._resolve_run(run_id)
        path = run_dir / "reports" / report_name
        if not path.exists():
            raise KeyError(report_name)
        return path.read_text(encoding="utf-8", errors="replace")

    def _iter_run_dirs(self) -> list[Path]:
        if not self.runs_root.exists():
            return []
        runs: list[Path] = []
        for entry in self.runs_root.iterdir():
            if not entry.is_dir():
                continue
            if entry.name in {"consistency-probe", "cache"}:
                continue
            if (entry / "trajectories").is_dir():
                runs.append(entry)
        return sorted(runs)

    def _resolve_run(self, run_id: str) -> Path:
        _validate_child_id(run_id)
        requested = self.runs_root / run_id
        if not requested.exists() or requested.parent.resolve() != self.runs_root:
            raise KeyError(run_id)
        return requested.resolve()

    def _resolve_trajectory(self, run_dir: Path, traj_id: str) -> Path:
        traj_dir = (run_dir / "trajectories" / traj_id).resolve()
        if not traj_dir.exists() or run_dir not in traj_dir.parents:
            raise KeyError(traj_id)
        return traj_dir

    def _resolve_harness_dir(self, run_dir: Path, harness_id: str) -> Path:
        if "->" in harness_id:
            raise KeyError(harness_id)
        _validate_child_id(harness_id)
        harness_root = (run_dir / "harness").resolve()
        requested = harness_root / harness_id
        if not requested.is_dir() or requested.parent.resolve() != harness_root:
            raise KeyError(harness_id)
        return requested.resolve()

    def _summarize_run(self, run_dir: Path) -> dict[str, Any]:
        manifest = self._read_manifest(run_dir)
        summary = self._read_json(run_dir / "reports" / "summary.json")
        config = self._read_json(run_dir / "config.json")
        usage = self._read_usage_summary(run_dir)
        final_val = summary.get("final_val") if isinstance(summary, dict) else {}
        return {
            "id": run_dir.name,
            "name": run_dir.name,
            "path": str(run_dir),
            "dataset_spec": config.get("dataset_spec"),
            "round_count": len(manifest.get("round_dirs", [])),
            "trajectory_count": manifest.get("trajectory_count", 0),
            "trajectory_counts_by_kind": manifest.get("trajectory_counts_by_kind", {}),
            "trajectory_counts_by_stage": manifest.get("trajectory_counts_by_stage", {}),
            "final_val": final_val,
            "initial_harness_id": summary.get("initial_harness_id"),
            "final_harness_id": summary.get("final_harness_id"),
            "end_timestamp": summary.get("end_timestamp"),
            "usage": usage.get("overall", {}),
        }

    def _read_manifest(self, run_dir: Path) -> dict[str, Any]:
        path = run_dir / "reports" / "manifest.json"
        data = self._read_json(path, default=None)
        if isinstance(data, dict):
            return data
        trajectory_metas = [self._read_json(path) for path in sorted((run_dir / "trajectories").glob("*/meta.json"))]
        counts_by_kind = Counter()
        counts_by_stage = Counter()
        trajectory_ids: list[str] = []
        for meta in trajectory_metas:
            if not isinstance(meta, dict):
                continue
            trajectory_ids.append(str(meta.get("id") or ""))
            counts_by_kind[str(meta.get("kind") or "(none)")] += 1
            counts_by_stage[str(meta.get("stage") or "(none)")] += 1
        return {
            "run_dir": str(run_dir),
            "report_files": sorted(
                path.relative_to(run_dir).as_posix()
                for path in (run_dir / "reports").glob("*")
                if path.is_file()
            ),
            "round_dirs": sorted(
                path.relative_to(run_dir).as_posix()
                for path in (run_dir / "rounds").glob("round_*")
                if path.is_dir()
            ),
            "trajectory_count": len(trajectory_ids),
            "trajectory_ids": trajectory_ids,
            "trajectory_counts_by_kind": dict(sorted(counts_by_kind.items())),
            "trajectory_counts_by_stage": dict(sorted(counts_by_stage.items())),
            "generated_fallback": True,
        }

    def _read_usage_summary(self, run_dir: Path) -> dict[str, Any]:
        path = run_dir / "reports" / "usage_summary.json"
        data = self._read_json(path, default=None)
        if isinstance(data, dict):
            return data
        overall = Counter()
        by_kind: dict[str, Counter[str]] = {}
        by_stage: dict[str, Counter[str]] = {}
        with_usage_count = 0
        for meta_path in sorted((run_dir / "trajectories").glob("*/meta.json")):
            meta = self._read_json(meta_path, default={})
            traj_dir = meta_path.parent
            events = load_events(traj_dir / "events.jsonl")
            usage = _extract_usage(events)
            kind = str(meta.get("kind") or "(none)")
            stage = str(meta.get("stage") or "(none)")
            kind_bucket = by_kind.setdefault(kind, Counter())
            stage_bucket = by_stage.setdefault(stage, Counter())
            kind_bucket["trajectory_count"] += 1
            stage_bucket["trajectory_count"] += 1
            if usage is None:
                continue
            with_usage_count += 1
            overall["with_usage_count"] += 1
            kind_bucket["with_usage_count"] += 1
            stage_bucket["with_usage_count"] += 1
            for key, value in usage.items():
                overall[key] += value
                kind_bucket[key] += value
                stage_bucket[key] += value
        overall["trajectory_count"] = sum(1 for _ in (run_dir / "trajectories").glob("*/meta.json"))
        return {
            "overall": {key: int(value) for key, value in overall.items()},
            "by_kind": {key: {k: int(v) for k, v in value.items()} for key, value in sorted(by_kind.items())},
            "by_stage": {key: {k: int(v) for k, v in value.items()} for key, value in sorted(by_stage.items())},
            "with_usage_count": with_usage_count,
            "generated_fallback": True,
        }

    def _read_report_files(self, run_dir: Path) -> list[dict[str, Any]]:
        reports = []
        report_dir = run_dir / "reports"
        if not report_dir.exists():
            return reports
        for path in sorted(report_dir.glob("*")):
            if not path.is_file():
                continue
            reports.append(
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                }
            )
        return reports

    def _read_round_summary(self, run_dir: Path, round_dir: Path) -> dict[str, Any]:
        round_ix = int(round_dir.name.split("_", 1)[1])
        config = self._read_json(run_dir / "config.json", default={})
        optimize_candidates = self._read_json(round_dir / "optimize_candidates.json", default={})
        samples = optimize_candidates.get("samples", []) if isinstance(optimize_candidates, dict) else []
        unique_candidates = (
            optimize_candidates.get("unique_candidates", [])
            if isinstance(optimize_candidates, dict)
            else []
        )
        return {
            "round_ix": round_ix,
            "strategy": config.get("optimize_strategy", "diagnosis"),
            "input_harness_id": self._read_text(round_dir / "input_harness_id"),
            "candidate_harness_id": self._read_text(round_dir / "candidate_harness_id"),
            "accepted": _parse_bool_text(self._read_text(round_dir / "accepted")),
            "mean_score": _parse_float_text(self._read_text(round_dir / "mean_score")),
            "score_count": len(self._read_json(round_dir / "scores.json", default=[])),
            "optimize_samples": len(samples),
            "unique_candidate_count": len(unique_candidates),
            "winner_sample_index": (
                optimize_candidates.get("winner_sample_index")
                if isinstance(optimize_candidates, dict)
                else None
            ),
        }

    def _read_trajectory_summary(self, run_dir: Path, traj_id: str | None) -> dict[str, Any] | None:
        if not traj_id:
            return None
        traj_dir = run_dir / "trajectories" / traj_id
        if not traj_dir.exists():
            return {
                "id": traj_id,
                "missing": True,
            }
        meta = self._read_json(traj_dir / "meta.json", default={})
        return {
            "id": traj_id,
            "kind": meta.get("kind"),
            "task_id": meta.get("task_id"),
            "harness_id": meta.get("harness_id"),
            "stage": meta.get("stage"),
            "round_ix": meta.get("round_ix"),
            "sample_index": meta.get("sample_index"),
            "exit_code": meta.get("exit_code"),
            "timed_out": meta.get("timed_out"),
            "wall_time_s": meta.get("wall_time_s"),
            "model": meta.get("model"),
            "reasoning_effort": meta.get("reasoning_effort"),
            "cache_mode": meta.get("cache_mode"),
        }

    def _find_task_diagnosis(self, run_dir: Path, task_id: str) -> dict[str, Any] | None:
        for path in sorted((run_dir / "rounds").glob("round_*/diagnoses.json")):
            diagnoses = self._read_json(path, default=[])
            if not isinstance(diagnoses, list):
                continue
            for diagnosis in diagnoses:
                if isinstance(diagnosis, dict) and diagnosis.get("task_id") == task_id:
                    return diagnosis
        return None

    def _task_selection(self, run_dir: Path, task_id: str) -> dict[str, Any] | None:
        selection = self._read_json(run_dir / "selection.json", default=None)
        if not isinstance(selection, dict):
            return None
        selected_task_ids = selection.get("selected_task_ids")
        selected = isinstance(selected_task_ids, list) and task_id in selected_task_ids
        difficulty_scores = selection.get("difficulty_scores") if isinstance(selection.get("difficulty_scores"), dict) else {}
        fingerprints = selection.get("fingerprints") if isinstance(selection.get("fingerprints"), dict) else {}
        return {
            "difficulty_score": difficulty_scores.get(task_id),
            "fingerprint": fingerprints.get(task_id),
            "selected": selected,
        }

    def _read_candidate_harness_diff(self, run_dir: Path, round_dir: Path) -> str:
        diff_path = round_dir / "candidate_harness_diff.patch"
        if diff_path.exists():
            return diff_path.read_text(encoding="utf-8", errors="replace")
        base_id = self._read_text(round_dir / "input_harness_id")
        candidate_id = self._read_text(round_dir / "candidate_harness_id")
        if not base_id or not candidate_id or candidate_id.startswith("("):
            return "(candidate harness diff unavailable)\n"
        before_dir = run_dir / "harness" / base_id
        after_dir = run_dir / "harness" / candidate_id
        if not before_dir.exists() or not after_dir.exists():
            return "(candidate harness diff unavailable)\n"
        return _build_directory_diff(before_dir, after_dir)

    def _read_json(self, path: Path, default: Any = _MISSING) -> Any:
        if not path.exists():
            return {} if default is _MISSING else default
        return json.loads(path.read_text(encoding="utf-8"))

    def _read_text(self, path: Path) -> str | None:
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8", errors="replace").strip()


def _validate_child_id(value: str) -> None:
    if value in {"", ".", ".."} or "/" in value or "\\" in value:
        raise KeyError(value)


def _extract_usage(events: list[dict[str, Any]]) -> dict[str, int] | None:
    usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    saw_usage = False
    for event in events:
        if event.get("type") != "turn.completed":
            continue
        payload = event.get("usage")
        if not isinstance(payload, dict):
            continue
        usage["input_tokens"] += int(payload.get("input_tokens", 0))
        usage["cached_input_tokens"] += int(payload.get("cached_input_tokens", 0))
        usage["output_tokens"] += int(payload.get("output_tokens", 0))
        saw_usage = True
    return usage if saw_usage else None


_STAGE_ORDER = {
    "round_solve_before": 0,
    "round_diagnose": 1,
    "round_optimize": 2,
    "round_solve_after": 3,
    "round_evaluate": 4,
    "final_val_grade": 5,
    "cli_val_grade": 5,
}


def _sort_stages(stages: set[str]) -> list[str]:
    return sorted(stages, key=lambda stage: (_STAGE_ORDER.get(stage, 99), stage))


def _trajectory_sort_key(trajectory: dict[str, Any]) -> tuple[int, int, int, str]:
    round_ix = trajectory.get("round_ix")
    sample_index = trajectory.get("sample_index")
    stage = trajectory.get("stage")
    return (
        round_ix if isinstance(round_ix, int) else 10_000,
        _STAGE_ORDER.get(stage, 99) if isinstance(stage, str) else 99,
        sample_index if isinstance(sample_index, int) else 10_000,
        str(trajectory.get("id") or ""),
    )


def _read_text_excerpt(path: Path, *, max_chars: int) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[:max_chars]


def _ungraded_score() -> dict[str, Any]:
    return {
        "kind": "ungraded",
        "score": None,
        "ctrf": None,
        "reward": None,
        "rationale": None,
        "source": "ungraded",
    }


def _read_consistency_score(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("value")
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    rationale = payload.get("rationale")
    return {
        "kind": "consistency",
        "score": value,
        "ctrf": None,
        "reward": None,
        "rationale": rationale if isinstance(rationale, str) else None,
        "source": "final_message.txt",
    }


def _find_ctrf_score(grades: list[Any], traj_id: str, *, source: str) -> dict[str, Any] | None:
    for entry in grades:
        if not isinstance(entry, dict) or entry.get("trajectory_id") != traj_id:
            continue
        details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
        ctrf = details.get("ctrf_summary") if isinstance(details, dict) else None
        reward = details.get("reward") if isinstance(details, dict) else None
        score = entry.get("score")
        return {
            "kind": "ctrf",
            "score": float(score) if isinstance(score, int | float) and not isinstance(score, bool) else None,
            "ctrf": ctrf if isinstance(ctrf, dict) else None,
            "reward": str(reward) if reward is not None else None,
            "rationale": None,
            "source": source,
        }
    return None


def _read_grades_from_run_log(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "[":
            continue
        candidate = "\n".join(lines[index:])
        try:
            payload, _ = json.JSONDecoder().raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        return payload if isinstance(payload, list) else []
    return []


def _optimize_traj_ids(round_dir: Path, optimize_candidates: Any) -> list[str]:
    ids: list[str] = []
    if isinstance(optimize_candidates, dict):
        for sample in optimize_candidates.get("samples", []):
            if not isinstance(sample, dict):
                continue
            traj_id = sample.get("optimize_traj_id")
            if isinstance(traj_id, str) and traj_id not in ids:
                ids.append(traj_id)
    if ids:
        return ids

    ids_payload = round_dir / "optimize_traj_ids.json"
    if ids_payload.exists():
        raw_ids = json.loads(ids_payload.read_text(encoding="utf-8"))
        if isinstance(raw_ids, list):
            return [traj_id for traj_id in raw_ids if isinstance(traj_id, str)]

    legacy_id = (round_dir / "optimize_traj_id").read_text(encoding="utf-8").strip() if (round_dir / "optimize_traj_id").exists() else ""
    return [legacy_id] if legacy_id else []


def _build_directory_diff(before_dir: Path, after_dir: Path) -> str:
    patch_lines: list[str] = []
    before_files = {
        path.relative_to(before_dir).as_posix(): path
        for path in before_dir.rglob("*")
        if path.is_file()
    }
    after_files = {
        path.relative_to(after_dir).as_posix(): path
        for path in after_dir.rglob("*")
        if path.is_file()
    }
    for rel in sorted(set(before_files) | set(after_files)):
        before_path = before_files.get(rel)
        after_path = after_files.get(rel)
        before_text = before_path.read_text(encoding="utf-8", errors="replace") if before_path else ""
        after_text = after_path.read_text(encoding="utf-8", errors="replace") if after_path else ""
        if before_text == after_text:
            continue
        diff = difflib.unified_diff(
            before_text.splitlines(keepends=True),
            after_text.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
        )
        patch_lines.extend(diff)
    return "".join(patch_lines) if patch_lines else "(no diff)\n"


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _parse_bool_text(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _parse_float_text(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None
