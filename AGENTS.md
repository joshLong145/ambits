# Agent Instructions

## Non-Interactive Shell Commands

**ALWAYS use non-interactive flags** with file operations to avoid hanging on confirmation prompts.

Shell commands like `cp`, `mv`, and `rm` may be aliased to include `-i` (interactive) mode on some systems, causing the agent to hang indefinitely waiting for y/n input.

**Use these forms instead:**
```bash
# Force overwrite without prompting
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file

# For recursive operations
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest
```

**Other commands that may prompt:**
- `scp` - use `-o BatchMode=yes` for non-interactive
- `ssh` - use `-o BatchMode=yes` to fail instead of prompting
- `apt-get` - use `-y` flag
- `brew` - use `HOMEBREW_NO_AUTO_UPDATE=1` env var

## Documentation

The README is a short overview. Reference docs live in the
[GitHub wiki](https://github.com/joshLong145/ambits/wiki), whose source is
`docs/wiki/` in this repo — edit there, never in the wiki UI. When a change
alters a command, flag, output format or config key, update the matching
`docs/wiki/` page in the same change.

To publish, copy `docs/wiki/*.md` into a clone of
`git@github.com:joshLong145/ambits.wiki.git` and push it.

## Landing the Plane (Session Completion)

**When ending a work session**, complete ALL steps below. Work is NOT complete until `git push` succeeds.

1. **Run quality gates** (if code changed) — what CI runs:
   ```bash
   cargo build
   cargo test
   ```
2. **Update docs** — `docs/wiki/` for any user-visible change, and publish the wiki if it changed.
3. **Push to remote**:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
4. **Clean up** — clear stashes, prune remote branches.
5. **Hand off** — summarize what changed and what is left for the next session.

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- If push fails, resolve and retry until it succeeds
