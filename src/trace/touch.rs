//! Which traces touched a file or symbol: the link from the tree to the
//! trace view. For the inspector's "these prompts read or wrote this".

use std::collections::HashMap;

use super::{SpanKind, Trace};
use crate::symbols::{nested_in, split_id};
use crate::write_index::WriteIndex;

/// One trace's calls on a file or symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Touch {
    /// The trace: its root span (a prompt, or a call before any).
    pub root: usize,
    /// Its first call on the target.
    pub span: usize,
    pub read: bool,
    pub wrote: bool,
    /// How many of its calls touched the target.
    pub calls: usize,
}

/// The traces whose calls touched `file` (project-relative) — or, given a
/// symbol id, that symbol — in time order.
///
/// A read touches a symbol when a symbol it was credited with reading is the
/// symbol, inside it, or around it — which counts the `ambits show`s and
/// searches that name no file of their own. A read credited with nothing
/// (of a file with no symbols, say) falls back to the file it named and the
/// symbol it targeted, if any. A write touches it when
/// its attribution names it or something nested in it (so a file-level
/// write touches the file only).
pub fn touches(trace: &Trace, file: &str, symbol: Option<&str>, writes: &WriteIndex) -> Vec<Touch> {
    let tree = trace.tree();
    let root_of = super::roots_by_span(&tree);
    let name = symbol.map(|id| split_id(id).1);
    let mut by_root: Vec<Touch> = Vec::new();
    let mut index: HashMap<usize, usize> = HashMap::new();

    for (i, s) in trace.spans().iter().enumerate() {
        let on_file = s.file.as_deref() == Some(file);
        // What the call read of it, as credited — which also finds the
        // `ambits show`s and searches that name no file of their own.
        let mut credited = s.reads_in(file).map(|(read, _)| read).peekable();
        let read = match name {
            _ if credited.peek().is_none() => {
                // Nothing credited: a read of the file still read it (one of
                // a file the tree has no symbols for, say).
                matches!(s.kind, SpanKind::Read(_)) && on_file && name.is_none_or(|name| s.symbol_name().is_none_or(|t| nested_in(name, &t) || nested_in(&t, name)))
            }
            None => true,
            Some(name) => credited.any(|read| nested_in(name, read) || nested_in(read, name)),
        };
        let wrote = on_file
            && s.kind == SpanKind::Write
            && match symbol {
                Some(id) => s.id.as_deref().and_then(|op| writes.get(op)).is_some_and(|w| w.touches_symbol(id)),
                None => true,
            };
        // A call on the file that neither read nor wrote it — a search, say —
        // touches the file, not a symbol in it.
        if !read && !wrote && !(on_file && symbol.is_none()) {
            continue;
        }
        let root = root_of.get(&i).copied().unwrap_or(i);
        match index.get(&root) {
            Some(&t) => {
                let touch = &mut by_root[t];
                touch.read |= read;
                touch.wrote |= wrote;
                touch.calls += 1;
            }
            None => {
                index.insert(root, by_root.len());
                by_root.push(Touch { root, span: i, read, wrote, calls: 1 });
            }
        }
    }
    by_root.sort_by_key(|t| (trace.spans()[t.root].start, t.root));
    by_root
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::{Effect, Prompt, ToolFinished};
    use std::path::Path;
    use std::sync::Arc;

    fn call(t: &mut Trace, id: &str, tool: &str, symbol: Option<&str>, at: &str, write: bool) {
        let mut c = crate::helpers::tool_call(tool, "/p/src/a.rs", crate::tracking::ReadDepth::FullBody);
        c.agent_id = Arc::from("main");
        c.tool_use_id = Some(Arc::from(id));
        c.timestamp_str = at.into();
        c.target_symbol = symbol.map(String::from);
        if write {
            c.effect = Effect::Write;
            c.read_depth = crate::tracking::ReadDepth::Unseen;
        }
        t.start(&c, Path::new("/p"));
        t.finish(&ToolFinished { id: Arc::from(id), agent_id: Arc::from("main"), timestamp: at.into(), error: false, message: None, child_agent: None, shown: Vec::new() });
    }

    fn prompt(t: &mut Trace, at: &str, text: &str) {
        t.prompt(&Prompt { agent_id: Arc::from("main"), timestamp: at.into(), text: text.into() });
    }

    /// An `ambits show` names no file of its own; what it was credited with
    /// reading is what it touched.
    #[test]
    fn a_show_touches_the_symbols_it_was_credited_with() {
        let mut t = Trace::default();
        prompt(&mut t, "2026-09-27T10:00:00Z", "first");
        let mut c = crate::helpers::tool_call("Bash", "/p/x", crate::tracking::ReadDepth::FullBody);
        c.file_path = None;
        c.agent_id = Arc::from("main");
        c.tool_use_id = Some(Arc::from("b1"));
        c.timestamp_str = "2026-09-27T10:00:01Z".into();
        t.start(&c, Path::new("/p"));
        t.note_read("b1", vec![("src/a.rs::App/run".into(), crate::tracking::ReadDepth::FullBody)]);
        let writes = WriteIndex::default();
        assert_eq!(touches(&t, "src/a.rs", Some("src/a.rs::App"), &writes).len(), 1, "App/run is inside App");
        assert_eq!(touches(&t, "src/a.rs", None, &writes).len(), 1);
        assert!(touches(&t, "src/a.rs", Some("src/a.rs::Other"), &writes).is_empty());

        let index = crate::trace::summary::TraceIndex::new(&t);
        let d = crate::trace::summary::detail(&t, &index, 0).unwrap();
        assert_eq!(d.files.iter().map(|f| f.file.as_str()).collect::<Vec<_>>(), vec!["src/a.rs"], "its file is in the summary");
        assert_eq!(d.files[0].symbols_read(&t), vec![(Some("App/run".into()), Some(crate::tracking::ReadDepth::FullBody))]);
    }

    #[test]
    fn a_symbol_is_touched_by_the_prompts_that_read_or_wrote_it() {
        let mut t = Trace::default();
        prompt(&mut t, "2026-09-27T10:00:00Z", "first"); // span 0
        call(&mut t, "r1", "Read", Some("impl App/fn run"), "2026-09-27T10:00:01Z", false); // 1
        call(&mut t, "r2", "Read", Some("Other"), "2026-09-27T10:00:02Z", false); // 2
        prompt(&mut t, "2026-09-27T11:00:00Z", "second"); // 3
        call(&mut t, "w1", "Edit", None, "2026-09-27T11:00:01Z", true); // 4
        call(&mut t, "r3", "Read", None, "2026-09-27T11:00:02Z", false); // 5

        let mut writes = WriteIndex::default();
        writes.insert(crate::writes::WriteRecord {
            op: "w1".into(),
            file: "src/a.rs".into(),
            level: crate::writes::Level::Symbol,
            syms: vec![("src/a.rs::App/run".into(), "b3:x".into())],
            ..Default::default()
        });

        let got = touches(&t, "src/a.rs", Some("src/a.rs::App/run"), &writes);
        assert_eq!(
            got,
            vec![
                Touch { root: 0, span: 1, read: true, wrote: false, calls: 1 },
                Touch { root: 3, span: 4, read: true, wrote: true, calls: 2 },
            ]
        );
        let parent = touches(&t, "src/a.rs", Some("src/a.rs::App"), &writes);
        assert_eq!(parent.len(), 2, "a read or write inside App touches App");
        assert!(touches(&t, "src/a.rs", Some("src/a.rs::Unrelated"), &writes).iter().all(|x| x.root == 3 && !x.wrote), "only the whole-file read");
        assert_eq!(touches(&t, "src/a.rs", None, &writes).iter().map(|x| x.calls).collect::<Vec<_>>(), vec![2, 2]);
        assert!(touches(&t, "src/b.rs", None, &writes).is_empty());
    }
}
