# MetaForge

请参阅 [MetaForge 文档索引](docs/metaforge/README.md)，了解仓库导览、从测试到补丁的源码路径，以及原生工具和技能清单。

---

<p align="center"><strong>Codex CLI</strong> 是 OpenAI 推出的编程智能体，可在你的计算机上本地运行。
<p align="center">
  <img src="https://github.com/openai/codex/blob/main/.github/codex-cli-splash.png" alt="Codex CLI 展示图" width="80%" />
</p>
</br>
如果你希望在代码编辑器（VS Code、Cursor、Windsurf）中使用 Codex，请<a href="https://developers.openai.com/codex/ide">安装 IDE 扩展</a>。
</br>如果你希望使用桌面应用，请运行 <code>codex app</code>，或访问 <a href="https://chatgpt.com/codex?app-landing-page=true">Codex 应用页面</a>。
</br>如果你要使用 OpenAI 的<em>云端智能体</em> <strong>Codex Web</strong>，请访问 <a href="https://chatgpt.com/codex">chatgpt.com/codex</a>。</p>

---

## 快速开始

### 安装并运行 Codex CLI

在 Mac 或 Linux 上运行以下命令，安装 Codex CLI：

```shell
curl -fsSL https://chatgpt.com/codex/install.sh | sh
```

在 Windows 上运行以下命令，安装 Codex CLI：

```shell
powershell -ExecutionPolicy ByPass -c "irm https://chatgpt.com/codex/install.ps1 | iex"
```

独立安装程序默认从 `https://releases.openai.com/codex` 下载；如果元数据或安装资源无法下载，则回退到 GitHub Releases。若要强制使用 GitHub Releases，请将 `CODEX_INSTALLER_USE_RELEASES_OPENAI_COM` 设置为 `false`，也可以使用 `0` 或 `no`：

```shell
curl -fsSL https://chatgpt.com/codex/install.sh | CODEX_INSTALLER_USE_RELEASES_OPENAI_COM=false sh
```

```powershell
$env:CODEX_INSTALLER_USE_RELEASES_OPENAI_COM='false'; irm https://chatgpt.com/codex/install.ps1 | iex
```

也可以通过以下包管理器安装 Codex CLI：

```shell
# 使用 npm 安装
npm install -g @openai/codex
```

```shell
# 使用 Homebrew 安装
brew install --cask codex
```

安装完成后，运行 `codex` 即可开始使用。

<details>
<summary>你也可以访问 <a href="https://github.com/openai/codex/releases/latest">GitHub 最新发布页面</a>，下载适合当前平台的可执行文件。</summary>

每个 GitHub 发布版本都包含多种可执行文件，通常选择以下一种即可：

- macOS
  - Apple Silicon/arm64: `codex-aarch64-apple-darwin.tar.gz`
  - x86_64（较旧的 Mac 硬件）：`codex-x86_64-apple-darwin.tar.gz`
- Linux
  - x86_64: `codex-x86_64-unknown-linux-musl.tar.gz`
  - arm64: `codex-aarch64-unknown-linux-musl.tar.gz`

每个压缩包仅包含一个文件，其名称中带有平台信息，例如 `codex-x86_64-unknown-linux-musl`。解压后，通常可以将它重命名为 `codex`。

</details>

## 文档

- [**Codex 文档**](https://developers.openai.com/codex)
- [**贡献指南**](./docs/contributing.md)
- [**安装与构建**](./docs/install.md)
- [**开源基金**](./docs/open-source-fund.md)

本仓库采用 [Apache-2.0 许可证](LICENSE)。
