# MetaForge Documentation

MetaForge is a research fork of [openai/codex](https://github.com/openai/codex), maintained under DeepSoftwareAnalytics. This directory contains English editions of the project's earlier Chinese source notes.

## Reading guide

| Document | What it covers |
| --- | --- |
| [Repository and harness guide](CODEX-HARNESS-GUIDE.md) | Repository layout, the Rust agent loop, execution, context, and persistence |
| [Test-to-patch source map](CODEX-HARNESS-PATHS.md) | Test execution, output processing, subsequent model requests, file changes, and trace evidence |
| [Native tools and skills](CODEX-NATIVE-TOOLS-AND-SKILLS.md) | Implemented tool families, availability conditions, embedded skills, and discovery mechanisms |

Start with the repository guide, follow one command through the source map, then use the tools and skills inventory as a reference.

## Source provenance and version boundaries

| Item | Reference |
| --- | --- |
| Upstream repository | `https://github.com/openai/codex` |
| Fork repository | `https://github.com/DeepSoftwareAnalytics/MetaForge` |
| Initial fork of upstream `main`, checked October 29, 2026 | `a3697361dfc0bc438b65ae8a9aa0e598ecd45251` |
| Source snapshot reviewed by the translated guides | `c260087efe8cc6dcc0526e7018ae9e0e3fc134e3` |
| Original guide review dates | October 27–29, 2026 |
| English edition | October 29, 2026 |

The guides do not claim that every path, feature default, or behavior is unchanged at the fork's current `main`. Their source links are pinned to the reviewed upstream commit so historical claims remain inspectable.

The English editions preserve the original technical scope while replacing machine-specific absolute links with portable references. Local installation observations are labeled as such; they are not universal Codex defaults. The upstream README already exists in English and is retained in the root README after the MetaForge introduction.

## Source code, installed programs, and external harnesses

These are separate artifacts:

- **Source:** Rust and supporting files in this repository.
- **Executable:** A compiled Codex program. Editing source does not modify an already installed executable.
- **External harness:** Instructions, skills, templates, or scripts loaded at runtime. Updating these usually does not require rebuilding Codex, provided they are not compiled-in resources.

For a Rust tool or agent-loop change to affect an experiment, build a compatible executable and explicitly configure the experiment to use it. Installing an upstream release does not automatically build or run this fork.

The earlier local RHO experiment used a prebuilt Codex CLI 0.163.0 binary. Neither the source snapshot reviewed by these guides nor this fork's initial `main` should be silently substituted for that executable when reporting experimental provenance.

## Repository boundaries

The initial documentation update adds no credentials, local configuration, build outputs, or raw experiment trajectories. The original Chinese notes remain in their local source repositories. The English guides here cover Codex source structure; the separate RHO experiment reports have not been imported as part of this update.

Upstream licensing and notices remain in [LICENSE](../../LICENSE) and [NOTICE](../../NOTICE).
