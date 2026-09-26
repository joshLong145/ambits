use std::fs;
use std::io::Write;
use std::path::PathBuf;

use color_eyre::eyre::{Result, WrapErr};

const SKILL_MD: &str = include_str!("../skills/ambit/SKILL.md");
const COVERAGE_GUIDE: &str = include_str!("../skills/ambit/coverage-guide.md");
const SESSION_MGMT: &str = include_str!("../skills/ambit/session-management.md");
const EXAMPLES: &str = include_str!("../skills/ambit/examples.md");

const SKILL_FILES: &[(&str, &str)] = &[
    ("SKILL.md", SKILL_MD),
    ("coverage-guide.md", COVERAGE_GUIDE),
    ("session-management.md", SESSION_MGMT),
    ("examples.md", EXAMPLES),
];

pub fn install(global: bool, project: Option<PathBuf>) -> Result<()> {
    let mut out = std::io::stdout().lock();
    let target_dir = if global {
        let home = std::env::var("HOME")
            .wrap_err("HOME environment variable not set")?;
        PathBuf::from(home).join(".claude/skills/ambit")
    } else if let Some(p) = project {
        p.join(".claude/skills/ambit")
    } else {
        PathBuf::from(".claude/skills/ambit")
    };

    fs::create_dir_all(&target_dir)
        .wrap_err_with(|| format!("Failed to create directory: {}", target_dir.display()))?;

    for (filename, content) in SKILL_FILES {
        let path = target_dir.join(filename);
        fs::write(&path, content)
            .wrap_err_with(|| format!("Failed to write {}", path.display()))?;
    }

    let scope = if global {
        "globally (all projects)"
    } else {
        "for this project"
    };

    writeln!(out, "Installed ambit skill {} to:", scope)?;
    writeln!(out, "  {}", target_dir.display())?;
    writeln!(out)?;
    writeln!(out, "Use /ambit in Claude Code to check coverage.")?;

    Ok(())
}
