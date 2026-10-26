# Getting Started

## Install

```bash
cargo install ambits
```

Or download a prebuilt binary from
[Releases](https://github.com/joshLong145/ambits/releases): static Linux builds
for x86_64 and aarch64, and macOS builds for Apple Silicon and Intel. Each is a
`.tar.gz` named `ambits-<version>-<target>`, with `SHA256SUMS` to check it
against. Next to each archive is `ambits-<version>-<target>.cdx.json`, a
[CycloneDX](https://cyclonedx.org/) 1.5 SBOM of the crates compiled into that
binary (build-time dependencies are listed with scope `excluded`), for feeding
to a vulnerability or license scanner.

```bash
tar -xzf ambits-0.25.0-aarch64-apple-darwin.tar.gz
mv ambits-0.25.0-aarch64-apple-darwin/ambits ~/.local/bin/
ambits --version
```

Or build from source — Rust 1.88+ (`rust-version` in `Cargo.toml`):

```bash
cargo build --release
cargo test
```

## Wire it into Claude Code

Two one-time steps, both safe to re-run.

```bash
# Hand read history back automatically after every compaction
ambits hook install --project .

# Teach the agent when to check its coverage and how to fetch definitions
ambits skill install --global
```

`hook install` merges a `SessionStart` hook into `.claude/settings.json` (or
`~/.claude/settings.json` with `--global`); a settings file that cannot be
parsed is left untouched and the snippet is printed instead. See
[Restoring Context](Restoring-Context#automatic-hand-back-after-compaction).

`skill install` writes the skill to `~/.claude/skills/ambit/` with `--global`,
or to `.claude/skills/ambit/` in the current project (or the one named by
`--project`).

## First session

Start the TUI in the project the agent is working in. It tails the session log
and — while it runs — keeps the read journal that `restore-context` answers
from.

```bash
ambits -p .
```

Then, from the agent or another terminal:

```bash
ambits -p . rg 'Journal::open'                     # search, attributed to symbols
ambits -p . callers centered_rect                  # who calls this?
ambits -p . show 'src/app.rs::App/process_compaction'  # read one symbol
ambits -p . restore-context                        # what has this session read?
ambits -p . --coverage                             # how much of the project is that?
```

`-p/--project` is optional: without it, the project is the nearest enclosing
directory containing `.git` or `.ambits`, so `ambits rg foo` works from
anywhere inside a repository. PATH arguments still resolve against the
current directory, and with none the whole project is searched.

## Supported languages

| Backend | Languages |
|---|---|
| tree-sitter (default) | Rust, Python, TypeScript, Java, Markdown (headings as symbols) |
| [Serena](https://github.com/oraios/serena) (`--serena`) | Anything Serena's language servers support |

Every text file is still searchable with `rg`/`grep`; a file no parser handles
simply has no symbols.
