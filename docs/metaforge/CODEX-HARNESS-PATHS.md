# Codex Harness Source Map for Test-to-Patch Research

Original review: October 28, 2026. English edition: October 29, 2026.

Reviewed source: `c260087efe8cc6dcc0526e7018ae9e0e3fc134e3`.

This document translates the original machine-specific absolute-path index into a portable source map. All source links refer to the reviewed upstream commit; paths in the tables are relative to the Codex repository root. For a local absolute path, prepend the location of your own checkout.

The reading order is conceptual, not a strict serial call chain. History and persistence span execution, while model requests and tool operations alternate under core's control. These implementation details describe this Codex snapshot and are not evidence about other agent harnesses.

## 1. Overall reading map

```text
cli / tui / exec                 User entry points and interaction
        ↓
app-server                      Sessions, requests, and events
        ↓
core                            Agent loop and context management
        ├── codex-api            Model requests and responses
        └── core/src/tools       Parse, route, and handle tool calls
                ├── exec-server Execute commands and access files
                └── apply-patch Parse and apply patches

Cross-cutting components:
history / context_manager       History types and model context
rollout / thread-store          Session persistence
rollout-trace                   Structured execution traces
```

## 2. Modules

| Module | Purpose | Source path |
| --- | --- | --- |
| `cli` / `codex-cli` | Rust CLI entry point and subcommand dispatch | [codex-rs/cli](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/cli) |
| `tui` / `codex-tui` | Terminal interface, input, and result display | [codex-rs/tui](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/tui) |
| `exec` / `codex-exec` | Complete noninteractive agent tasks, including JSONL output | [codex-rs/exec](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/exec) |
| `app-server` | Application requests, threads, turns, and events | [codex-rs/app-server](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/app-server) |
| `core` | Agent loop, context, and tool scheduling | [codex-rs/core](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core) |
| `codex-api` | Model service API and request/response events | [codex-rs/codex-api](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/codex-api) |
| `core/src/tools` | Core routing, handlers, runtimes, and result processing; not a separate crate | [codex-rs/core/src/tools](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools) |
| `exec-server` | Local/remote processes, files, and execution environments | [codex-rs/exec-server](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/exec-server) |
| `apply-patch` | Patch parsing and file changes | [codex-rs/apply-patch](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/apply-patch) |
| `utils/output-truncation` | Byte/token limits for command and tool output | [codex-rs/utils/output-truncation](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/utils/output-truncation) |
| `history` | Domain types for model history and persisted rollouts | [codex-rs/history](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/history) |
| `rollout` | Session file writing, discovery, and reading | [codex-rs/rollout](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/rollout) |
| `thread-store` | Persistence interfaces keyed by thread ID | [codex-rs/thread-store](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/thread-store) |
| `rollout-trace` | Structured trace formats, writers, and event reduction | [codex-rs/rollout-trace](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/rollout-trace) |

The root [codex-cli/](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-cli) directory is npm packaging and startup code, distinct from the Rust CLI package. Likewise, `exec` runs an agent task, while `exec-server` provides lower-level execution capabilities.

## 3. How are test commands executed and collected?

Follow the tool entry point into its runtime and process management.

| Location | What to inspect | Source |
| --- | --- | --- |
| Routing | How a model call reaches its handler | [router.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/router.rs) |
| Command handler | Parsing and execution of `exec_command` and `write_stdin` | [handlers/unified_exec.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/handlers/unified_exec.rs) |
| Orchestration | Approvals, sandbox selection, and retry conditions | [orchestrator.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/orchestrator.rs) |
| Execution runtime | Connecting calls to execution mechanisms | [runtimes/unified_exec.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/runtimes/unified_exec.rs) |
| Process management | Starting, continuing, feeding, and collecting processes | [process_manager.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/unified_exec/process_manager.rs) |
| Process object | Process state and output handling | [process.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/unified_exec/process.rs) |
| Execution service | Local/remote process and filesystem interfaces | [exec-server/src/lib.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/exec-server/src/lib.rs) |

Distinguish startup, continued execution, subsequent output retrieval, and final completion. A tool returning once does not imply the test process has finished.

## 4. Which parts of test output reach the model?

| Location | What to inspect | Source |
| --- | --- | --- |
| Tool output and formatting | `ExecCommandToolOutput`, model output policy, exit codes, process IDs, and truncation notices | [tools/context.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/context.rs) |
| General truncation | Byte/token budgets | [output-truncation/src/lib.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/utils/output-truncation/src/lib.rs) |
| Process output buffer | Head/tail retention during collection | [head_tail_buffer.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/unified_exec/head_tail_buffer.rs) |

A field named `raw_output` may already have been limited during collection or buffering. Terminal display, saved traces, and model-visible output must be checked separately.

## 5. How does feedback enter the next request?

| Location | What to inspect | Source |
| --- | --- | --- |
| Turn loop | Model requests, tool operations, and continuation | [session/turn.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/session/turn.rs) |
| Streaming events | Dispatch of tool calls emitted by the model | [stream_events_utils.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/stream_events_utils.rs) |
| Context management | History normalization and updates | [context_manager/](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/context_manager) |
| History implementation | How tool results enter model context | [context_manager/history.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/context_manager/history.rs) |
| Core model client | Session requests and connections | [client.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/client.rs) |
| Model API | Request/response types and API capabilities | [codex-api/src/lib.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/codex-api/src/lib.rs) |

Search `turn.rs` for `run_turn`, `drain_in_flight`, `record_annotated_conversation_items`, `build_prompt`, and `try_run_sampling_request`. Their source order does not necessarily match execution order.

To make a causal claim, establish that test feedback reached subsequent requests and account for truncation, compaction, and intervening feedback. A test followed by an edit is only temporal ordering, not sufficient causal evidence.

## 6. How do generated patches change files?

| Location | What to inspect | Source |
| --- | --- | --- |
| Patch handler | Receiving and interpreting patch calls | [handlers/apply_patch.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/handlers/apply_patch.rs) |
| Patch runtime | Execution and permission integration | [runtimes/apply_patch.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/runtimes/apply_patch.rs) |
| Patch library | Data structures and application interfaces | [apply-patch/src/lib.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/apply-patch/src/lib.rs) |
| Parser | Text to file operations and hunks | [parser.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/apply-patch/src/parser.rs) |
| File update | Constructing new content from hunks | [file_update.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/apply-patch/src/file_update.rs) |

Shell commands and scripts can also edit files. Study the actual workspace diff and other writes, not just `apply_patch` calls.

## 7. What does the saved trajectory contain?

| Location | What to inspect | Source |
| --- | --- | --- |
| History types | Shared representation for model history and rollouts | [history/src/lib.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/history/src/lib.rs) |
| Rollout payloads | Persisted content types | [rollout_payload.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/history/src/rollout_payload.rs) |
| Recorder | Writing session records | [rollout/src/recorder.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/rollout/src/recorder.rs) |
| Persistence policy | Which items qualify for persistence | [rollout/src/policy.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/rollout/src/policy.rs) |
| Thread store | Persistence boundary keyed by thread ID | [thread-store/src/store.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/thread-store/src/store.rs) |
| Live thread | Persistence operations for active threads | [live_thread.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/thread-store/src/live_thread.rs) |
| Local backend | Local creation, reads, history, and rollouts | [thread-store/src/local/](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/thread-store/src/local) |
| Trace API | Structured formats and public interfaces | [rollout-trace/src/lib.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/rollout-trace/src/lib.rs) |
| Trace writer | Writing trace events | [writer.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/rollout-trace/src/writer.rs) |
| Trace reduction | Reducing events to session, reasoning, and tool states | [reducer/](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/rollout-trace/src/reducer) |

A trajectory file is not a filesystem snapshot and is not necessarily the complete input of every model request. Its contents depend on recording formats, enabled features, and persistence policy.

## 8. Shortest useful reading sequence

1. Read `turn.rs` to understand model/tool iteration.
2. Read Unified Exec handlers and the process manager to understand test execution.
3. Inspect tool output formatting and truncation to determine visible feedback.
4. Trace history updates and subsequent model requests.
5. Follow patch handlers and application code into file changes.
6. Compare recorders and persistence policy with the evidence available offline.

For the broader repository map, see [Codex Harness Guide](CODEX-HARNESS-GUIDE.md).
