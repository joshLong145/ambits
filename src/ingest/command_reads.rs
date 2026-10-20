//! The file lines a shell command prints: `sed -n 10,40p a.rs`, `head -n 20
//! a.rs`, `tail -n 5 a.rs`, `cat a.rs`, `nl a.rs`, alone or in a chain of
//! `;`, `&&` and `||`, after any `cd`. A command it cannot read with
//! certainty (a variable, a glob, a heredoc, a script it does not know)
//! names nothing, so it earns no credit rather than a guessed one.

use std::ops::Range;
use std::path::{Path, PathBuf};

/// Which lines of a file a command printed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Lines {
    /// 1-based and half-open; `end` is `u32::MAX` for "to the end".
    Range(Range<u32>),
    /// The last `n` lines, which depend on the file's length.
    Tail(u32),
    Whole,
}

/// One file a command printed lines of. `path` is as the command named it,
/// joined to any directory an earlier `cd` in the chain moved to; a relative
/// one is relative to the shell's working directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileRead {
    pub path: PathBuf,
    pub lines: Lines,
}

/// The files `command` printed lines of, in order.
pub fn file_reads(command: &str) -> Vec<FileRead> {
    let Some(tokens) = tokenize(command) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut dir = PathBuf::new();
    for pipeline in tokens.split(|t| matches!(t, Token::Sep)) {
        // Only a pipeline's first command reads files; the rest read its output.
        let Some(first) = pipeline.split(|t| matches!(t, Token::Pipe)).next() else {
            continue;
        };
        let Some(words) = first.iter().map(Token::word).collect::<Option<Vec<&str>>>() else {
            continue;
        };
        match words.as_slice() {
            ["cd", to] => dir = dir.join(to),
            ["cd", ..] => return out,
            [program, args @ ..] => {
                out.extend(reads_of(program, args).into_iter().map(|r| FileRead {
                    path: dir.join(r.path),
                    ..r
                }))
            }
            [] => {}
        }
    }
    out
}

fn reads_of(program: &str, args: &[&str]) -> Vec<FileRead> {
    let (lines, files) = match program {
        "sed" => match sed(args) {
            Some(found) => found,
            None => return Vec::new(),
        },
        "head" | "tail" => match count(args) {
            Some((n, from_start, files)) => {
                let lines = match (program, from_start) {
                    ("head", _) => Lines::Range(1..n.saturating_add(1)),
                    (_, true) => Lines::Range(n.max(1)..u32::MAX),
                    _ => Lines::Tail(n),
                };
                (lines, files)
            }
            None => return Vec::new(),
        },
        "cat" | "nl" => {
            // Flags that take no value; `nl`'s all take one, so none is followed.
            let plain: &[&str] = if program == "cat" {
                &["-n", "-b", "-A", "-v", "-e", "-t", "-s", "-E", "-T"]
            } else {
                &[]
            };
            if args
                .iter()
                .any(|a| a.starts_with('-') && !plain.contains(a))
            {
                return Vec::new();
            }
            (
                Lines::Whole,
                args.iter()
                    .copied()
                    .filter(|a| !a.starts_with('-'))
                    .collect(),
            )
        }
        _ => return Vec::new(),
    };
    files
        .into_iter()
        .filter(|f| plain_path(f))
        .map(|f| FileRead {
            path: PathBuf::from(f),
            lines: lines.clone(),
        })
        .collect()
}

/// `sed -n 'A,Bp' FILE`, `sed -n 'Ap' FILE`, `sed -n 'A,$p' FILE`: one file,
/// one print command.
fn sed<'a>(args: &[&'a str]) -> Option<(Lines, Vec<&'a str>)> {
    let mut quiet = false;
    let mut script = None;
    let mut files = Vec::new();
    let mut it = args.iter();
    while let Some(&a) = it.next() {
        match a {
            "-n" | "--quiet" | "--silent" => quiet = true,
            "-e" => script = Some(*it.next()?),
            _ if a.starts_with('-') => return None,
            _ if script.is_none() => script = Some(a),
            _ => files.push(a),
        }
    }
    if !quiet || files.len() != 1 {
        return None;
    }
    let body = script?.strip_suffix('p')?;
    let (from, to) = match body.split_once(',') {
        Some((a, "$")) => (a.parse::<u32>().ok()?, u32::MAX),
        Some((a, b)) => (a.parse().ok()?, b.parse::<u32>().ok()?.checked_add(1)?),
        None => {
            let a: u32 = body.parse().ok()?;
            (a, a.checked_add(1)?)
        }
    };
    (from >= 1 && from < to).then_some((Lines::Range(from..to), files))
}

/// `head`/`tail`'s line count and files: `-n N`, `-nN`, `-N`, `--lines=N`,
/// `-n +K` (from line K, for `tail`); 10 when not given.
fn count<'a>(args: &[&'a str]) -> Option<(u32, bool, Vec<&'a str>)> {
    let mut n = None;
    let mut files = Vec::new();
    let mut it = args.iter();
    while let Some(&a) = it.next() {
        let value = if a == "-n" {
            Some(*it.next()?)
        } else if let Some(v) = a.strip_prefix("--lines=") {
            Some(v)
        } else if let Some(v) = a.strip_prefix("-n") {
            Some(v)
        } else if a.len() > 1 && a.starts_with('-') && a[1..].bytes().all(|b| b.is_ascii_digit()) {
            Some(&a[1..])
        } else if a.starts_with('-') {
            return None;
        } else {
            files.push(a);
            None
        };
        if let Some(v) = value {
            n = Some(v);
        }
    }
    let (from_start, digits) = match n {
        Some(v) => v.strip_prefix('+').map_or((false, v), |rest| (true, rest)),
        None => (false, "10"),
    };
    Some((digits.parse().ok()?, from_start, files)).filter(|(_, _, files)| !files.is_empty())
}

/// A path that names one file as written: no variable, glob, `~` or stdin.
fn plain_path(p: &str) -> bool {
    !p.is_empty() && p != "-" && !p.starts_with('~') && !p.contains(['$', '*', '?', '[', '{', '`'])
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Word(String),
    /// `;`, `&&`, `||`, `&` or a newline: the next command runs on its own.
    Sep,
    Pipe,
}

impl Token {
    fn word(&self) -> Option<&str> {
        match self {
            Token::Word(w) => Some(w),
            _ => None,
        }
    }
}

/// Shell words and operators, quoting honoured, redirections dropped with
/// their targets. `None` for a heredoc, a subshell or a command
/// substitution, whose text no simple reading can follow.
fn tokenize(command: &str) -> Option<Vec<Token>> {
    let mut out = Vec::new();
    let mut word = String::new();
    let mut in_word = false;
    let mut chars = command.chars().peekable();
    let flush = |word: &mut String, in_word: &mut bool, out: &mut Vec<Token>| {
        if *in_word {
            out.push(Token::Word(std::mem::take(word)));
            *in_word = false;
        }
    };
    while let Some(c) = chars.next() {
        match c {
            '\'' => {
                in_word = true;
                for q in chars.by_ref() {
                    if q == '\'' {
                        break;
                    }
                    word.push(q);
                }
            }
            '"' => {
                in_word = true;
                while let Some(q) = chars.next() {
                    match q {
                        '"' => break,
                        '\\' => word.push(chars.next()?),
                        '$' | '`' => return None,
                        _ => word.push(q),
                    }
                }
            }
            '\\' => {
                in_word = true;
                match chars.next()? {
                    '\n' => {}
                    q => word.push(q),
                }
            }
            '`' => return None,
            '$' if chars.peek() == Some(&'(') => return None,
            ';' | '\n' => {
                flush(&mut word, &mut in_word, &mut out);
                out.push(Token::Sep);
            }
            '&' if chars.peek() == Some(&'&') => {
                chars.next();
                flush(&mut word, &mut in_word, &mut out);
                out.push(Token::Sep);
            }
            '|' if chars.peek() == Some(&'|') => {
                chars.next();
                flush(&mut word, &mut in_word, &mut out);
                out.push(Token::Sep);
            }
            '|' => {
                flush(&mut word, &mut in_word, &mut out);
                out.push(Token::Pipe);
            }
            '<' if chars.peek() == Some(&'<') => return None,
            '(' | ')' => return None,
            '&' => {
                flush(&mut word, &mut in_word, &mut out);
                out.push(Token::Sep);
            }
            '<' | '>' => {
                // `2>`: the descriptor is part of the redirection, not a word.
                if in_word && !word.is_empty() && word.bytes().all(|b| b.is_ascii_digit()) {
                    word.clear();
                    in_word = false;
                }
                flush(&mut word, &mut in_word, &mut out);
                // A redirection's target is not a file the command printed.
                while chars.peek().is_some_and(|c| matches!(c, '>' | '&' | '|')) {
                    chars.next();
                }
                while chars
                    .peek()
                    .is_some_and(|c| c.is_whitespace() && *c != '\n')
                {
                    chars.next();
                }
                while chars
                    .peek()
                    .is_some_and(|c| !c.is_whitespace() && !matches!(c, ';' | '|' | '&'))
                {
                    chars.next();
                }
            }
            c if c.is_whitespace() => flush(&mut word, &mut in_word, &mut out),
            c => {
                in_word = true;
                word.push(c);
            }
        }
    }
    flush(&mut word, &mut in_word, &mut out);
    Some(out)
}

/// A line-numbered search a command ran (`grep -n`, `rg -n`): its output
/// names the file lines it printed, which the result credits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Search {
    /// The directory it ran in, from any `cd` before it; a relative one is
    /// relative to the shell's working directory, as for [`FileRead`].
    pub dir: PathBuf,
    /// The one path it searched, for a lone search whose output may carry
    /// no file names, so every `N:` line is that file's if it is one.
    pub file: Option<PathBuf>,
}

/// The line-numbered searches `command` ran, in order.
pub fn searches(command: &str) -> Vec<Search> {
    let Some(tokens) = tokenize(command) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut dir = PathBuf::new();
    let mut commands = 0;
    for pipeline in tokens.split(|t| matches!(t, Token::Sep)) {
        let Some(first) = pipeline.split(|t| matches!(t, Token::Pipe)).next() else {
            continue;
        };
        let Some(words) = first.iter().map(Token::word).collect::<Option<Vec<&str>>>() else {
            continue;
        };
        match words.as_slice() {
            ["cd", to] => dir = dir.join(to),
            ["cd", ..] => return out,
            [program, args @ ..] => {
                commands += 1;
                if let Some(file) = search_of(program, args) {
                    out.push(Search {
                        dir: dir.clone(),
                        file: file.map(|f| dir.join(f)),
                    });
                }
            }
            [] => {}
        }
    }
    // With other commands' output beside it, an unprefixed `N:` line could
    // be any of them.
    if commands > 1 {
        for search in &mut out {
            search.file = None;
        }
    }
    out
}

/// For `grep`/`rg` given line numbers: `Some(Some(file))` when it searched
/// one file and prints no file names, `Some(None)` when it names them.
fn search_of<'a>(program: &str, args: &[&'a str]) -> Option<Option<&'a str>> {
    let takes_value: &[char] = match program {
        "grep" | "egrep" | "fgrep" => &['e', 'f', 'A', 'B', 'C', 'm', 'd', 'D'],
        "rg" => &[
            'e', 'f', 'g', 't', 'T', 'A', 'B', 'C', 'm', 'M', 'j', 'E', 'r',
        ],
        _ => return None,
    };
    let (mut numbered, mut names_files, mut pattern_given) = (false, false, false);
    let mut operands = Vec::new();
    let mut it = args.iter();
    while let Some(&a) = it.next() {
        if a == "--" {
            operands.extend(it.by_ref().copied());
            break;
        }
        if let Some(long) = a.strip_prefix("--") {
            match long.split('=').next().unwrap_or_default() {
                "line-number" => numbered = true,
                "recursive" | "with-filename" | "files-with-matches" | "count" => {
                    names_files = true
                }
                "regexp" | "file" => {
                    pattern_given = true;
                    if !long.contains('=') {
                        it.next();
                    }
                }
                "no-filename" | "no-heading" => {}
                other
                    if !long.contains('=')
                        && matches!(
                            other,
                            "after-context"
                                | "before-context"
                                | "context"
                                | "max-count"
                                | "glob"
                                | "type"
                        ) =>
                {
                    it.next();
                }
                _ => {}
            }
            continue;
        }
        if let Some(flags) = a.strip_prefix('-').filter(|f| !f.is_empty()) {
            for (i, c) in flags.char_indices() {
                match c {
                    'n' => numbered = true,
                    'r' | 'R' | 'H' | 'l' | 'L' | 'c' if program != "rg" || c != 'r' => {
                        names_files = true
                    }
                    _ if takes_value.contains(&c) => {
                        if matches!(c, 'e' | 'f') {
                            pattern_given = true;
                        }
                        if i + c.len_utf8() == flags.len() {
                            it.next();
                        }
                        break;
                    }
                    _ => {}
                }
            }
            continue;
        }
        operands.push(a);
    }
    if !numbered {
        return None;
    }
    let files: Vec<&str> = operands
        .into_iter()
        .skip(usize::from(!pattern_given))
        .collect();
    Some(match files.as_slice() {
        [file] if !names_files && plain_path(file) && !file.ends_with('/') => Some(*file),
        _ => None,
    })
}

/// The `(file, line)` pairs a search's `output` printed: `path:N:` and
/// context `path-N-` lines, or `N:`/`N-` lines of a search's one file. A
/// path may itself hold `-` or `:`, so every split is returned and the
/// caller keeps those naming a file it knows; relative paths are joined to
/// the search's directory.
pub fn printed_lines(output: &str, searches: &[Search]) -> Vec<(PathBuf, u32)> {
    let mut out = std::collections::BTreeSet::new();
    let lone = match searches {
        [Search {
            file: Some(file), ..
        }] => Some(file),
        _ => None,
    };
    for line in output.lines().take(20_000) {
        // A lone search's argument may be a directory, whose output names
        // files after all: both readings are offered.
        if let Some(file) = lone {
            if let Some(n) = leading_number(line) {
                out.insert((file.clone(), n));
            }
        }
        for (i, c) in line.char_indices() {
            if (c == ':' || c == '-') && i > 0 {
                if let Some(n) = leading_number(&line[i + 1..]) {
                    let path = Path::new(&line[..i]);
                    for search in searches {
                        out.insert((search.dir.join(path), n));
                    }
                }
            }
        }
    }
    out.into_iter().collect()
}

/// `N` from `N:...` or `N-...`.
fn leading_number(s: &str) -> Option<u32> {
    let digits = s.bytes().take_while(u8::is_ascii_digit).count();
    let rest = s.as_bytes().get(digits)?;
    (digits > 0 && matches!(rest, b':' | b'-'))
        .then(|| s[..digits].parse().ok())
        .flatten()
}

/// `path` resolved against `cwd` when relative.
pub fn resolve(path: &Path, cwd: Option<&Path>) -> PathBuf {
    match cwd {
        Some(cwd) if path.is_relative() => cwd.join(path),
        _ => path.to_path_buf(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reads(command: &str) -> Vec<(String, Lines)> {
        file_reads(command)
            .into_iter()
            .map(|r| (r.path.to_string_lossy().into_owned(), r.lines))
            .collect()
    }

    #[test]
    fn sed_prints_a_range_a_line_or_to_the_end() {
        assert_eq!(
            reads("sed -n 10,40p src/a.rs"),
            [("src/a.rs".into(), Lines::Range(10..41))]
        );
        assert_eq!(
            reads("sed -n '7p' a.rs"),
            [("a.rs".into(), Lines::Range(7..8))]
        );
        assert_eq!(
            reads("sed -n -e '5,$p' a.rs"),
            [("a.rs".into(), Lines::Range(5..u32::MAX))]
        );
    }

    #[test]
    fn sed_that_edits_or_does_not_print_a_range_names_nothing() {
        assert!(reads("sed -i 's/a/b/' a.rs").is_empty());
        assert!(
            reads("sed 's/a/b/' a.rs").is_empty(),
            "without -n it prints everything, edited"
        );
        assert!(
            reads("sed -n '/fn/p' a.rs").is_empty(),
            "a pattern address: lines unknown"
        );
        assert!(
            reads("sed -n 1,5p a.rs b.rs").is_empty(),
            "line numbers run on across files"
        );
    }

    #[test]
    fn head_and_tail_in_their_spellings() {
        assert_eq!(
            reads("head -n 20 a.rs"),
            [("a.rs".into(), Lines::Range(1..21))]
        );
        assert_eq!(reads("head -5 a.rs"), [("a.rs".into(), Lines::Range(1..6))]);
        assert_eq!(reads("head a.rs"), [("a.rs".into(), Lines::Range(1..11))]);
        assert_eq!(reads("tail -n 30 a.rs"), [("a.rs".into(), Lines::Tail(30))]);
        assert_eq!(
            reads("tail -n +100 a.rs"),
            [("a.rs".into(), Lines::Range(100..u32::MAX))]
        );
        assert_eq!(
            reads("head --lines=3 a.rs b.rs").len(),
            2,
            "each file's first lines"
        );
    }

    #[test]
    fn cat_and_nl_print_the_whole_file() {
        assert_eq!(
            reads("cat a.rs b.rs"),
            [("a.rs".into(), Lines::Whole), ("b.rs".into(), Lines::Whole)]
        );
        assert_eq!(
            reads("nl -b a.rs").len(),
            0,
            "an option that takes a value is not followed"
        );
        assert_eq!(reads("cat -n a.rs"), [("a.rs".into(), Lines::Whole)]);
    }

    #[test]
    fn a_chain_is_read_command_by_command_after_any_cd() {
        let got = reads("cd /ws/repo && sed -n 1,5p a.py; grep -n x b.py | head -3; cat 'c d.md'");
        assert_eq!(
            got,
            [
                ("/ws/repo/a.py".into(), Lines::Range(1..6)),
                ("/ws/repo/c d.md".into(), Lines::Whole)
            ]
        );
    }

    #[test]
    fn only_a_pipelines_first_command_reads_files() {
        assert_eq!(
            reads("cat a.rs | head -n 3"),
            [("a.rs".into(), Lines::Whole)]
        );
        assert!(reads("git show HEAD:a.rs | head -n 20").is_empty());
    }

    #[test]
    fn a_line_numbered_search_is_found_in_its_spellings() {
        let one = |c: &str| searches(c);
        assert_eq!(
            one("grep -n foo a.py"),
            [Search {
                dir: "".into(),
                file: Some("a.py".into())
            }]
        );
        assert_eq!(
            one("grep -rn 'class X' -A5 src/"),
            [Search {
                dir: "".into(),
                file: None
            }]
        );
        assert_eq!(
            one("rg --line-number x src"),
            [Search {
                dir: "".into(),
                file: Some("src".into())
            }],
            "rg names no file for one path argument"
        );
        assert_eq!(
            one("grep -e foo -n a.py b.py"),
            [Search {
                dir: "".into(),
                file: None
            }],
            "two files: names printed"
        );
        assert!(one("grep foo a.py").is_empty(), "no line numbers");
        assert!(
            one("grep -A5 n a.py").is_empty(),
            "the n is -A's value's neighbour, not a flag"
        );
        let chained = one("cd repo && grep -n x a.py; sed -n 1,2p b.py");
        assert_eq!(
            chained,
            [Search {
                dir: "repo".into(),
                file: None
            }],
            "another command's output beside it"
        );
    }

    #[test]
    fn printed_lines_come_from_matches_and_context() {
        let s = [Search {
            dir: "/ws".into(),
            file: None,
        }];
        let got = printed_lines(
            "docs/worker-mcp.md:57:a match\ndocs/worker-mcp.md-58-context\n--\nnot a hit",
            &s,
        );
        assert!(got.contains(&("/ws/docs/worker-mcp.md".into(), 57)));
        assert!(got.contains(&("/ws/docs/worker-mcp.md".into(), 58)));
        let lone = [Search {
            dir: "/ws".into(),
            file: Some("/ws/a.py".into()),
        }];
        assert_eq!(
            printed_lines("12:def f():\n13-    pass\nx", &lone),
            [("/ws/a.py".into(), 12), ("/ws/a.py".into(), 13)]
        );
    }

    #[test]
    fn what_cannot_be_read_with_certainty_names_nothing() {
        assert!(reads("cat $FILE").is_empty());
        assert!(reads("cat src/*.rs").is_empty());
        assert!(reads("cat ~/notes.md").is_empty());
        assert!(reads("cat \"$(ls)\"").is_empty());
        assert!(reads("cat > out.txt <<'EOF'\nx\nEOF").is_empty());
        assert_eq!(
            reads("cat a.rs > copy.rs"),
            [("a.rs".into(), Lines::Whole)],
            "a redirect target is not read"
        );
        assert_eq!(
            reads("cat a.rs 2>/dev/null"),
            [("a.rs".into(), Lines::Whole)]
        );
    }
}
