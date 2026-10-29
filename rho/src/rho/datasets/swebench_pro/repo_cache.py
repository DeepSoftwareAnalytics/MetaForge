from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from rho.datasets.swebench_pro.util import cache_root, file_lock, git, safe_name


class RepoCache:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or cache_root() / "swebench-pro" / "repos"
        self.root.mkdir(parents=True, exist_ok=True)

    def materialize(self, row: dict[str, Any], dest: Path) -> None:
        source_repo = row.get("repo_path")
        if source_repo:
            self._copy_local_repo(Path(str(source_repo)).expanduser().resolve(), dest)
            self._checkout(dest, str(row["base_commit"]))
            return

        repo = str(row["repo"])
        base_commit = str(row["base_commit"])
        cached = self._ensure_cached(repo, base_commit)
        if dest.exists():
            shutil.rmtree(dest)
        git(["clone", "--quiet", str(cached), str(dest)], timeout_s=1200)
        self._checkout(dest, base_commit)

    def _ensure_cached(self, repo: str, base_commit: str) -> Path:
        target = self.root / safe_name(repo)
        lock_path = self.root / f"{safe_name(repo)}.lock"
        with file_lock(lock_path):
            if not target.exists():
                git(
                    [
                        "clone",
                        "--quiet",
                        f"https://github.com/{repo}.git",
                        str(target),
                    ],
                    timeout_s=1800,
                )
            else:
                if not self._has_commit(target, base_commit):
                    git(["fetch", "--quiet", "origin"], cwd=target, timeout_s=1800)
            self._ensure_commit(target, base_commit)
        return target

    def _has_commit(self, repo_dir: Path, commit: str) -> bool:
        try:
            git(["cat-file", "-e", f"{commit}^{{commit}}"], cwd=repo_dir)
            return True
        except RuntimeError:
            return False

    def _ensure_commit(self, repo_dir: Path, commit: str) -> None:
        try:
            git(["cat-file", "-e", f"{commit}^{{commit}}"], cwd=repo_dir)
        except RuntimeError:
            git(["fetch", "--quiet", "origin", commit], cwd=repo_dir, timeout_s=1800)
            git(["cat-file", "-e", f"{commit}^{{commit}}"], cwd=repo_dir)
        git(["update-ref", f"refs/rho/base/{commit}", commit], cwd=repo_dir)

    def _copy_local_repo(self, source: Path, dest: Path) -> None:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest, symlinks=True)

    def _checkout(self, repo_dir: Path, commit: str) -> None:
        try:
            git(["checkout", "--quiet", "--detach", commit], cwd=repo_dir)
        except RuntimeError as checkout_error:
            try:
                git(["fetch", "--quiet", "origin", commit], cwd=repo_dir, timeout_s=1800)
            except RuntimeError:
                raise checkout_error
            git(["checkout", "--quiet", "--detach", commit], cwd=repo_dir)
        git(["reset", "--hard", "--quiet", commit], cwd=repo_dir)
        git(["clean", "-fdx", "--quiet"], cwd=repo_dir)
