# Codex CLI Native Tools and Skills: Source Inventory

Original review and English edition: October 29, 2026.

Reviewed source: `c260087efe8cc6dcc0526e7018ae9e0e3fc134e3`.

This is an English translation of a local source inventory. Source links are pinned to the reviewed commit. Observations about the author's installed skills are historical local observations, not requirements or defaults for other installations.

## 1. What does this source support?

This Codex source includes command execution, patch application, image viewing, planning, user interaction, multi-agent collaboration, MCP resource access, and Code Mode. Bundled extensions also implement web search, image generation, goals, skill access, memories, and related capabilities.

**Native support does not mean enabled by default or visible in every session.** The effective tool set depends on model capabilities, configuration, feature flags, permissions, execution environment, extensions, MCP servers, and host-provided dynamic tools. There is no single useful default tool count independent of these conditions.

This source snapshot embeds exactly five system skills:

1. `imagegen`: image generation and editing workflows.
2. `openai-docs`: official OpenAI and Codex information and usage guidance.
3. `review-agent`: read-only code review.
4. `skill-creator`: skill creation and maintenance.
5. `skill-installer`: skill installation.

The original review also confirmed files for all five under the local `CODEX_HOME/skills/.system/`. File presence alone does not establish the version of a running binary or prove that each skill was included in a session's visible catalog.

## 2. Tools, skills, plugins, and CLI commands

| Concept | Meaning | Example |
| --- | --- | --- |
| Tool | A model-callable operation with arguments, execution, and a result | `exec_command`, `apply_patch` |
| Skill | Instructions for a class of tasks, optionally including scripts, references, and templates | `skill-creator` |
| MCP tool | An operation supplied by a connected MCP service | A code-hosting service's PR lookup |
| Plugin | A distribution package for skills, MCP configuration, and related assets | Pages or Sites |
| CLI subcommand or slash command | A user-facing control surface, distinct from a model tool | `codex exec`, `/skills` |

A skill's entry point is `SKILL.md`, including its name, description, and instructions. The model generally receives catalog information first and reads the full instructions when needed. MCP provides data and operation interfaces. See the [official skills concept guide](https://developers.openai.com/plugins/concepts/skills).

Do not confuse:

- `exec_command`: executes a shell command.
- Code Mode `exec`: runs JavaScript that orchestrates tools.
- `codex exec`: starts a noninteractive agent task from a terminal.
- The `imagegen` skill: instructions for image tasks.
- The `image_gen.imagegen` tool: the actual generation/editing operation.

This document uses `namespace.tool` notation. Protocols, interfaces, and Code Mode may render names differently; V2 multi-agent namespaces are configurable.

## 3. How tools are assembled

| Location | Responsibility |
| --- | --- |
| [spec_plan.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/spec_plan.rs) | Assembles core, MCP, extension, and dynamic tools and determines exposure |
| [registry.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/registry.rs) | Tool implementations and exposure/permission information |
| [router.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/router.rs) | Dispatches model calls |
| [handlers/](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/handlers) | Core tool handlers |
| [ext/](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/ext) | Internal runtime extensions; not the user's installed-plugin directory |
| [features/src/lib.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/features/src/lib.rs) | Feature flags, stages, and source defaults |
| [code-mode-protocol/src/lib.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/code-mode-protocol/src/lib.rs) | Public Code Mode names such as `exec` and `wait` |

`build_tool_router()` registers core tools and adds MCP, extensions, and dynamic tools. Policy, tool search, and Code Mode affect the final exposure. Tools may be visible directly, discoverable later, or callable through Code Mode. Absence from the initial prompt does not prove absence from the implementation.

## 4. Core tools

### 4.1 Commands, file edits, images, and plans

| Tool | Purpose | Registration or availability conditions | Implementation |
| --- | --- | --- | --- |
| `exec_command` | Commands, tests, file reads, searches, and Git | Shell feature, model support, and applicable policies | [unified_exec.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/handlers/unified_exec.rs) |
| `write_stdin` | Send process input or retrieve subsequent output | Persistent-process Unified Exec path | [write_stdin.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/handlers/unified_exec/write_stdin.rs) |
| `apply_patch` | Add, change, or delete files | Nonempty model `apply_patch_tool_type`, plus tool policy | [apply_patch.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/handlers/apply_patch.rs) |
| `view_image` | Return an image to the model | `ViewImage` enabled; argument support also depends on model/environment | [view_image.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/handlers/view_image.rs) |
| `update_plan` | Maintain plan steps and status | `update_plan_enabled` | [plan.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/handlers/plan.rs) |

In this snapshot, `add_shell_tools()` can register a one-shot `exec_command` when Unified Exec is disabled but command execution is allowed. Do not assume older `shell`/`shell_command` fallback behavior applies here.

### 4.2 Interaction, permissions, context, and time

| Tool | Purpose | Main conditions | Implementation |
| --- | --- | --- | --- |
| `request_user_input` | Structured questions that wait for a response | Interaction enabled; allowed collaboration modes are determined separately | [request_user_input.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/handlers/request_user_input.rs) |
| `request_user_input_async` | Questions while work continues | Interaction enabled, root agent, model support | [request_user_input_async.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/handlers/request_user_input_async.rs) |
| `send_message_to_user_async` | Asynchronous user messaging | Root agent and applicable flags/model capability | [send_message_to_user_async.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/handlers/send_message_to_user_async.rs) |
| `request_permissions` | Request additional permissions | `RequestPermissionsTool` enabled; authorization remains policy-controlled | [request_permissions.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/handlers/request_permissions.rs) |
| `new_context` | Request a new context window | `TokenBudget` enabled | [new_context_window.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/handlers/new_context_window.rs) |
| `get_context_remaining` | Remaining-context information | `TokenBudget` enabled | [get_context_remaining.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/handlers/get_context_remaining.rs) |
| `clock.curr_time` | Current time | Time reminders enabled or model clock support | [current_time.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/handlers/current_time.rs) |
| `clock.sleep` | Wait for a duration | Sleep feature and mode conditions | [sleep.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/handlers/sleep.rs) |
| `wait_for_environment` | Wait for deferred environment preparation | `DeferredExecutor` enabled | [wait_for_environment.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/handlers/wait_for_environment.rs) |

### 4.3 MCP resources and discovery

| Tool | Purpose | Main conditions | Implementation |
| --- | --- | --- | --- |
| `list_mcp_resources` | List MCP resources | MCP servers present or effective mode is CodeModeOnly | [mcp_resource/](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/handlers/mcp_resource) |
| `list_mcp_resource_templates` | List parameterized resources | Same registration condition | [mcp_resource/](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/handlers/mcp_resource) |
| `read_mcp_resource` | Read a resource | Same registration condition; actual resource must be accessible | [mcp_resource/](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/handlers/mcp_resource) |
| `tool_search` | Discover deferred tools | Exposure mode, model capability, and policy | [tool_search.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/handlers/tool_search.rs) |
| `request_plugin_install` | Initiate plugin installation requests | ToolSuggest, Apps, and Plugins conditions, plus candidates | [request_plugin_install.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/handlers/request_plugin_install.rs) |
| `list_available_plugins_to_install` | List installable candidates | Above conditions and ListTool presentation mode | [list_available_plugins_to_install.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/handlers/list_available_plugins_to_install.rs) |

The first three are Codex's resource-access interfaces. Business operations registered by individual MCP servers are separate dynamic tools, not a fixed native inventory.

### 4.4 Multi-agent V1 and V2

V1 uses the `multi_agent_v1` namespace:

| Tool | Purpose |
| --- | --- |
| `multi_agent_v1.spawn_agent` | Create a child agent |
| `multi_agent_v1.send_input` | Send input to a child |
| `multi_agent_v1.resume_agent` | Resume a child |
| `multi_agent_v1.wait_agent` | Wait for state or results |
| `multi_agent_v1.close_agent` | Close a child |

See [multi_agents/](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/handlers/multi_agents) and [multi_agents_spec.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/handlers/multi_agents_spec.rs).

V2 uses these base names, with a host-configured namespace such as `collaboration`:

| Base name | Purpose |
| --- | --- |
| `spawn_agent` | Create a child agent |
| `send_message` | Message an existing agent |
| `followup_task` | Assign a follow-up task |
| `wait_agent` | Wait for updates |
| `interrupt_agent` | Interrupt current execution |
| `list_agents` | List agents and status |

See [multi_agents_v2/](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/handlers/multi_agents_v2). V1 has depth-related limits. V2 can disable native tools, direct messaging, or waiting and is also affected by model support and exposure mode. Implementation support does not authorize or require delegation for every task.

### 4.5 Code Mode

| Tool | Purpose | Condition |
| --- | --- | --- |
| `exec` | Execute JavaScript to compose or concurrently invoke tools and return results | Effective mode is CodeMode or CodeModeOnly |
| `wait` | Wait for a yielded Code Mode execution cell | Registered with Code Mode |

See [core/src/tools/code_mode/](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/code_mode) and [code-mode-runtime/](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/code-mode-runtime).

For example, JavaScript may call `tools.exec_command(...)`. The outer `exec` and an inner test command are different levels; outer execution success does not establish test success.

### 4.6 Test-only tool

`test_sync_tool`, implemented in [test_sync.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/handlers/test_sync.rs), is registered only when the model's experimental supported-tool list includes it. It supports harness testing/synchronization; it is not a general test runner for user projects.

## 5. Tools supplied by bundled extensions

These implementations are included in source rather than necessarily coming from installed third-party plugins. Runtime installation, configuration, and backend requirements still apply.

### 5.1 Search, images, goals, and cross-environment files

| Tool | Purpose | Conditions and implementation |
| --- | --- | --- |
| Hosted `web_search` tool type | Model-service web search | Provider support and search mode; standalone path not selected; [hosted_spec.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/core/src/tools/hosted_spec.rs) |
| `web.run` | Standalone search/browsing | Extension available and standalone path selected; [ext/web-search/](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/ext/web-search) |
| `image_gen.imagegen` | Image generation/editing | Extension, provider, and model conditions; [ext/image-generation/](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/ext/image-generation) |
| `get_goal` | Goal state, progress, and budget | Goal runtime visibility; [ext/goal/](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/ext/goal) |
| `create_goal` | Create a goal | Goal visibility and execution-state requirements |
| `update_goal` | Update goal status | Goal visibility and execution-state requirements |
| `multi_env_copy_file` | Copy between execution environments | Extension and environment support; [ext/multi-env-copy-file/](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/ext/multi-env-copy-file) |

Hosted search and `web.run` are distinct paths, not two search tools guaranteed to coexist. An image skill's presence does not establish service availability.

### 5.2 Skills and memories

| Tool | Purpose | Implementation |
| --- | --- | --- |
| `skills.list` | List available skill package/catalog information | [ext/skills/src/tools/](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/ext/skills/src/tools) |
| `skills.read` | Read instructions or package resources | Same directory |
| `skills.installations` | Query available cloud/environment installation locations; **does not install skills** | [installations.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/ext/skills/src/tools/installations.rs) |
| `memories.add_ad_hoc_note` | Add an ad hoc memory note | [ext/memories/src/tools/](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/ext/memories/src/tools) |
| `memories.list` | List memories | Same directory |
| `memories.read` | Read memory content | Same directory |
| `memories.search` | Search memories | Same directory |

Skill tools require the current step's catalog state to allow tool access; `installations` also requires discovery_v2. Memory tools require the feature, memory-use configuration, and dedicated-tools conditions.

File-based skills may also be read through command tools. Skill use does not always produce a `skills.read` call.

### 5.3 History and notes

Implementation: [ext/history-notes/src/tools.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/ext/history-notes/src/tools.rs).

| Tool | Purpose |
| --- | --- |
| `history.list_windows` | List historical context windows |
| `history.list_items` | List history items |
| `history.read_item` | Read one item |
| `history.search_contents` | Search history |
| `notes.list_files_by_prefix` | List notes by prefix |
| `notes.read_file` | Read a note |
| `notes.search_contents` | Search notes |
| `notes.append_to_file` | Append to a note |
| `notes.write_file` | Write a note |

Availability depends on history-note settings under token budgets, the provider, and authentication. Note paths have their own semantics and need not be local filesystem paths.

### 5.4 Agent message board

Implementation: [message-board/src/tools/selection.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/message-board/src/tools/selection.rs), integrated through [ext/agent-message-board/](https://github.com/openai/codex/tree/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/ext/agent-message-board).

| Base name | Purpose |
| --- | --- |
| `create_channel` | Create a channel |
| `get_channels` | Query channels |
| `list_threads` | List discussion threads |
| `search_posts` | Search messages |
| `read_thread` | Read a thread |
| `read_post` | Read a message |
| `subscribe` | Subscribe |
| `unsubscribe` | Unsubscribe |
| `post` | Publish a message |

A functioning message-board backend is required, and namespaces are host-configured. These tools are distinct from V2 point-to-point `send_message`.

## 6. Source feature defaults

The table reports selected `FeatureSpec.default_enabled` values in the reviewed source. It is not the user's actual configuration or a complete availability calculation.

| Flag | Source default | Qualification |
| --- | --- | --- |
| `shell_tool` | true | Model must also allow shell |
| `unified_exec` | true | Managed policy can affect the effective value |
| `unified_exec_tty` | true | Controls pseudo-terminal support |
| `view_image` | true | Tool policy and other conditions still apply |
| `sleep_tool` | true | Mode and clock conditions apply |
| `multi_agent` | true | Version and depth conditions still matter |
| `multi_agent_v2` | false | Does not prove V2 is absent in a particular session |
| `goals` | true | Runtime controls visibility and execution |
| `code_mode` / `code_mode_only` | false / false | Effective mode also considers model information |
| `code_mode_host` | true | Host support does not ensure `exec` is exposed |
| `request_permissions_tool` | false | Other approval mechanisms can still exist |
| `token_budget` | false | Affects context-related tools |
| `current_time_reminder` | false | Model clock support can still expose time tools |
| `memories` | false | Memory configuration and dedicated-tools conditions also apply |
| `multi_env_copy_file` | false | Extension and environment support required |

See [features/src/lib.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/features/src/lib.rs). For `apply_patch`, user questions, and planning, inspect their registration conditions rather than inferring behavior solely from this table.

## 7. The five embedded system skills

### 7.1 Why these are embedded rather than merely examples

[skills/src/lib.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/skills/src/lib.rs) uses `include_dir!` to embed `src/assets/samples`. `install_system_skills()` writes the embedded resources into `CODEX_HOME/skills/.system`. Despite the directory name `samples`, it serves as the embedded source in this snapshot.

A content fingerprint determines whether the installed system directory needs rewriting. Put custom skills in project or user locations instead of relying on edits to a managed system cache.

### 7.2 Source inventory

| Skill | Purpose | Source |
| --- | --- | --- |
| `imagegen` | Generation, editing, transparency, and related workflows; built-in image tools preferred, with a CLI path under specified conditions | [imagegen/SKILL.md](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/skills/src/assets/samples/imagegen/SKILL.md) |
| `openai-docs` | Official information for OpenAI/Codex products, APIs, configuration, and use | [openai-docs/SKILL.md](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/skills/src/assets/samples/openai-docs/SKILL.md) |
| `review-agent` | Read-only review of specified changes, emphasizing actionable defects | [review-agent/SKILL.md](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/skills/src/assets/samples/review-agent/SKILL.md) |
| `skill-creator` | Create or maintain appropriately scoped instructions and resources | [skill-creator/SKILL.md](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/skills/src/assets/samples/skill-creator/SKILL.md) |
| `skill-installer` | Install from a curated list or GitHub repository path | [skill-installer/SKILL.md](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/skills/src/assets/samples/skill-installer/SKILL.md) |

The review agent's [agents/openai.yaml](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/skills/src/assets/samples/review-agent/agents/openai.yaml) sets `allow_implicit_invocation: false`. It must not be treated as implicitly invocable solely because a task matches. This setting alone does not establish why a skill might be absent from a particular session catalog.

### 7.3 Installed locations in the original local review

The five files were found at:

```text
$CODEX_HOME/skills/.system/
├── imagegen/SKILL.md
├── openai-docs/SKILL.md
├── review-agent/SKILL.md
├── skill-creator/SKILL.md
└── skill-installer/SKILL.md
```

This portable notation replaces machine-specific absolute paths in the Chinese edition. It reports the reviewed installation, not the contents of every user's system.

Explicit mentions such as `$skill-creator` differ from implicit selection by description. A present skill can be disabled or omitted under catalog presentation limits. See the [official skill guide](https://learn.chatgpt.com/docs/build-skills).

## 8. Skills that were not embedded system skills

The original session also exposed these sources:

| Name or group | Source |
| --- | --- |
| `retro` | A user-installed skill under `CODEX_HOME/skills/retro/`, outside the embedded source directory |
| `pages:*` | Pages plugin |
| `sites:*` | Sites plugin |
| `work-pets:*` | Work Pets plugin |
| `plugin-management:plugin-management` | Plugin management package |

This is a historical session inventory, not a scan of every cached plugin or a list of recommended dependencies. An installable skill in an official curated repository is not necessarily embedded in the CLI. An official plugin is also not automatically an embedded system skill.

## 9. Where skills are loaded from

Source entry points: [host_roots.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/ext/skills/src/host_roots.rs) and [host_service.rs](https://github.com/openai/codex/blob/c260087efe8cc6dcc0526e7018ae9e0e3fc134e3/codex-rs/ext/skills/src/host_service.rs).

| Source | Path or mechanism | Ownership |
| --- | --- | --- |
| Embedded system skills | `$CODEX_HOME/skills/.system` | The five embedded skills above |
| User directory | `$HOME/.agents/skills` | User-maintained |
| Compatibility user directory | `$CODEX_HOME/skills` | Retained legacy location in this snapshot |
| Project directories | `.agents/skills` from the current directory toward the repository root | Project-maintained |
| Configuration-associated project directory | `skills` under a project configuration directory | Depends on configuration layers and access |
| Administrator directory | `skills` under the system configuration directory, commonly `/etc/codex/skills` | Administrator-managed |
| Plugins | Skill paths declared by plugins | Plugin distribution |
| Environments, cloud, and custom providers | Skill provider and catalog interfaces | May not have a locally readable path |

The [official skill guide](https://learn.chatgpt.com/docs/build-skills) explains discovery, `/skills`, explicit mentions, and on-demand loading. For exact behavior in this historical snapshot, use the pinned source.

## 10. Implications for test-to-patch research

### 10.1 Tests normally use general tools

This snapshot's ordinary registration does not define universal `run_pytest` or `generate_test` tools. A typical sequence is:

```text
model proposes validation
  → patch or command writes tests
  → exec_command runs pytest / python / a project test command
  → write_stdin retrieves additional output if execution continues
  → formatted/truncated output enters context
  → model chooses to edit source, edit tests, or validate further
```

`pytest`, `python`, `rg`, and `git` are programs in the execution environment, not separate native Codex tools merely because they appear in trajectories.

### 10.2 No embedded skill universally governs test generation

The five embedded skills do not include a universal unit-test-generation skill. Testing behavior can also depend on system/developer instructions, project `AGENTS.md`, user requests, model decisions, and additional installed skills.

`review-agent` is a reviewer, not a test generator. `update_plan` maintains task plans; it is not evidence of an embedded skill named `plan`.

### 10.3 Record these dimensions separately

| Dimension | Evidence to record |
| --- | --- |
| Tool availability | Tools visible or dynamically discoverable at the time |
| Actual invocation | Tool name, arguments, and any Code Mode nesting |
| Test execution | Command, completion state, exit status, and output |
| Model-visible feedback | Truncation, incremental results, and compaction |
| Skill use | Catalog presence versus actual instruction reads or references |
| Subsequent changes | Production code, tests, configuration, or other files |

This inventory cannot establish that earlier DeepSeek experiments used the same tools or skill mechanisms. Check each experiment's actual harness and request records.

For detailed feedback propagation, see [Codex Harness Source Map](CODEX-HARNESS-PATHS.md).
