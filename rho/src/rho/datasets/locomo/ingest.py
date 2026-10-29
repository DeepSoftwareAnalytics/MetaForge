"""Read ``locomo10.json`` and materialize the raw sessions into a harness
directory tree. See spec §6 and §7 for the data model and output format.

Important quirks (verified against snap-research/locomo@main):

- ``conversation`` is a flat dict with keys ``speaker_a``, ``speaker_b``,
  and pairs ``session_N`` / ``session_N_date_time``. **No session count
  field.** Enumerate keys matching ``^session_\\d+$`` and sort by integer
  suffix.
- Multimodal turns have ``img_url`` as a **list**. Rendered as text using
  ``query`` (reliable) + ``blip_caption`` (often wrong but harmless).
- ``observation`` / ``session_summary`` / ``event_summary`` are NOT read
  — they leak QA answers.
- ``adversarial_answer`` is the distractor, not the gold; category 5 has
  no ``answer`` key for 444 of 446 entries.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SESSION_KEY_RE = re.compile(r"^session_(\d+)$")


@dataclass(frozen=True)
class LocomoConversation:
    sample_id: str
    speakers: tuple[str, str]
    sessions: tuple["LocomoSession", ...]
    qa: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class LocomoSession:
    index: int
    date_time: str
    turns: tuple[dict[str, Any], ...]


def load_locomo(path: Path) -> tuple[LocomoConversation, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    conversations: list[LocomoConversation] = []
    for conv in raw:
        conv_data = conv["conversation"]
        speaker_a = conv_data["speaker_a"]
        speaker_b = conv_data["speaker_b"]
        sessions = _extract_sessions(conv_data)
        conversations.append(
            LocomoConversation(
                sample_id=conv["sample_id"],
                speakers=(speaker_a, speaker_b),
                sessions=sessions,
                qa=tuple(conv["qa"]),
            )
        )
    return tuple(conversations)


def _extract_sessions(conv_data: dict[str, Any]) -> tuple[LocomoSession, ...]:
    numbered: list[tuple[int, LocomoSession]] = []
    for key, value in conv_data.items():
        m = _SESSION_KEY_RE.fullmatch(key)
        if not m:
            continue
        idx = int(m.group(1))
        date_time = conv_data.get(f"session_{idx}_date_time", "")
        numbered.append(
            (
                idx,
                LocomoSession(
                    index=idx,
                    date_time=date_time,
                    turns=tuple(value),
                ),
            )
        )
    numbered.sort(key=lambda pair: pair[0])
    return tuple(sess for _, sess in numbered)


def write_harness_tree(
    conversations: tuple[LocomoConversation, ...],
    dest: Path,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for conv in conversations:
        conv_dir = dest / conv.sample_id
        conv_dir.mkdir(parents=True, exist_ok=True)
        (conv_dir / "INDEX.md").write_text(_render_index(conv), encoding="utf-8")
        for session in conv.sessions:
            session_path = conv_dir / f"session_{session.index:02d}.md"
            session_path.write_text(_render_session(session), encoding="utf-8")


def _render_index(conv: LocomoConversation) -> str:
    speaker_a, speaker_b = conv.speakers
    n_sessions = len(conv.sessions)
    first_idx = conv.sessions[0].index if conv.sessions else 1
    last_idx = conv.sessions[-1].index if conv.sessions else 1
    return (
        f"# {conv.sample_id}\n"
        "\n"
        f"Speakers: {speaker_a}, {speaker_b}.\n"
        f"Sessions: {n_sessions} files in this directory "
        f"(session_{first_idx:02d}.md .. session_{last_idx:02d}.md), "
        "each labeled with its original date-time header. Read them in order.\n"
        "\n"
        "You may freely reorganize, summarize, or add new files in this directory\n"
        "to make future questions about this conversation easier to answer.\n"
    )


def _render_session(session: LocomoSession) -> str:
    lines: list[str] = [f"# session_{session.index:02d} — {session.date_time}", ""]
    for turn in session.turns:
        lines.append(f"{turn['speaker']}: {turn['text']}")
        if "img_url" in turn:
            query = turn.get("query", "").strip()
            caption = turn.get("blip_caption", "").strip()
            if query or caption:
                if query and caption:
                    lines.append(f"  [image: {query} — {caption}]")
                else:
                    lines.append(f"  [image: {query or caption}]")
    lines.append("")
    return "\n".join(lines)
