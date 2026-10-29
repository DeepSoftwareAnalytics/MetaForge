from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from rho.webui.data import RunRepository


def create_app(runs_root: str | Path | None = None) -> FastAPI:
    repo_root = Path(__file__).resolve().parents[3]
    resolved_runs_root = Path(runs_root).resolve() if runs_root else repo_root / "runs"
    repository = RunRepository(resolved_runs_root)

    app = FastAPI(title="rho runs ui", version="0.1.0")
    app.state.repository = repository

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/runs")
    async def list_runs() -> list[dict]:
        return repository.list_runs()

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict:
        return _or_404(lambda: repository.get_run(run_id))

    @app.get("/api/runs/{run_id}/rounds/{round_ix}")
    async def get_round(run_id: str, round_ix: int) -> dict:
        return _or_404(lambda: repository.get_round(run_id, round_ix))

    @app.get("/api/runs/{run_id}/selection")
    async def get_selection(run_id: str) -> dict:
        return _or_404(lambda: repository.get_selection(run_id))

    @app.get("/api/runs/{run_id}/tasks")
    async def get_tasks(run_id: str) -> list[dict]:
        return _or_404(lambda: repository.get_run_tasks(run_id))

    @app.get("/api/runs/{run_id}/tasks/{task_id}")
    async def get_task(run_id: str, task_id: str) -> dict:
        return _or_404(lambda: repository.get_task(run_id, task_id))

    @app.get("/api/runs/{run_id}/harness/{harness_id}")
    async def get_harness(run_id: str, harness_id: str) -> dict:
        return _or_404(lambda: repository.get_harness(run_id, harness_id))

    @app.get("/api/runs/{run_id}/harness/{harness_id}/file/{rel_path:path}")
    async def get_harness_file(run_id: str, harness_id: str, rel_path: str) -> PlainTextResponse:
        text = _or_404(lambda: repository.get_harness_file_text(run_id, harness_id, rel_path))
        return PlainTextResponse(text)

    @app.get("/api/runs/{run_id}/trajectories/{traj_id}")
    async def get_trajectory(run_id: str, traj_id: str) -> dict:
        return _or_404(lambda: repository.get_trajectory(run_id, traj_id))

    @app.get("/api/runs/{run_id}/trajectories/{traj_id}/score")
    async def get_trajectory_score(run_id: str, traj_id: str) -> dict:
        return _or_404(lambda: repository.get_trajectory_score(run_id, traj_id))

    @app.get("/api/runs/{run_id}/trajectories/{traj_id}/trace")
    async def get_trace(run_id: str, traj_id: str) -> dict:
        return _or_404(lambda: repository.get_trace(run_id, traj_id))

    @app.get("/api/runs/{run_id}/trajectories/{traj_id}/trace/{step_id}/output")
    async def get_trace_output(
        run_id: str,
        traj_id: str,
        step_id: str,
        start_line: int = Query(default=0, ge=0),
        max_lines: int = Query(default=120, ge=1, le=500),
    ) -> dict:
        return _or_404(
            lambda: repository.get_trace_output_chunk(
                run_id,
                traj_id,
                step_id,
                start_line=start_line,
                max_lines=max_lines,
            )
        )

    @app.get("/api/runs/{run_id}/trajectories/{traj_id}/artifacts/{artifact_name}")
    async def get_artifact(run_id: str, traj_id: str, artifact_name: str) -> PlainTextResponse:
        text = _or_404(lambda: repository.get_artifact_text(run_id, traj_id, artifact_name))
        return PlainTextResponse(text)

    @app.get("/api/runs/{run_id}/trajectories/{traj_id}/workspace-diff/{rel_path:path}")
    async def get_workspace_diff(run_id: str, traj_id: str, rel_path: str) -> PlainTextResponse:
        text = _or_404(lambda: repository.get_workspace_diff_text(run_id, traj_id, rel_path))
        return PlainTextResponse(text)

    @app.get("/api/runs/{run_id}/reports/{report_name}")
    async def get_report(run_id: str, report_name: str) -> PlainTextResponse:
        text = _or_404(lambda: repository.get_report_text(run_id, report_name))
        return PlainTextResponse(text)

    static_dir = repo_root / "webui" / "dist"
    assets_dir = static_dir / "assets"
    index_path = static_dir / "index.html"
    if index_path.exists() and assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

        @app.get("/", include_in_schema=False)
        def serve_index() -> FileResponse:
            return FileResponse(index_path)

        @app.get("/{full_path:path}", include_in_schema=False)
        def serve_spa(full_path: str):
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not found")
            candidate = static_dir / full_path
            if candidate.exists() and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(static_dir / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        def static_missing() -> PlainTextResponse:
            return PlainTextResponse(
                "UI build missing. Run `npm install` and `npm run build` in `webui/`, then restart `rho ui`.",
                status_code=503,
            )

    return app


def _or_404(fn):
    try:
        return fn()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
