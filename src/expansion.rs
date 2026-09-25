//! Which tree rows are open.
//!
//! Files and symbols default in opposite directions. A file row starts
//! closed, so the tree opens as a list of files; a symbol row starts open, so
//! expanding a file shows its whole outline. Each is stored as the set of
//! exceptions to its default, which is what makes the default hold for rows
//! that did not exist at startup — a file created while the TUI runs, or one
//! a Serena re-scan brings in — without anything having to register them.
//!
//! Exceptions are keyed by id and never pruned: an editor's atomic save
//! arrives as a remove then a create, and forgetting the file in between
//! would snap it shut on every save.

use std::collections::HashSet;

/// Which default a row follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowKind {
    File,
    Symbol,
}

impl RowKind {
    /// The state a row of this kind has until the user changes it — the one
    /// place either default is written down.
    fn default_expanded(self) -> bool {
        match self {
            RowKind::File => false,
            RowKind::Symbol => true,
        }
    }
}

/// Rows the user has moved away from their kind's default, indexed by kind.
#[derive(Debug, Default)]
pub struct Expansion([HashSet<String>; 2]);

impl Expansion {
    pub fn is_expanded(&self, id: &str, kind: RowKind) -> bool {
        self.0[kind as usize].contains(id) != kind.default_expanded()
    }

    /// Returns whether the state changed, so callers redraw only when needed.
    pub fn set(&mut self, id: &str, kind: RowKind, expanded: bool) -> bool {
        let exceptions = &mut self.0[kind as usize];
        if expanded == kind.default_expanded() {
            exceptions.remove(id)
        } else {
            exceptions.insert(id.to_owned())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KINDS: [RowKind; 2] = [RowKind::File, RowKind::Symbol];

    #[test]
    fn files_default_closed_and_symbols_default_open() {
        let e = Expansion::default();
        assert!(!e.is_expanded("a.rs", RowKind::File));
        assert!(e.is_expanded("a.rs::f", RowKind::Symbol));
    }

    #[test]
    fn flipping_and_restoring_reports_each_change_and_stores_nothing() {
        for kind in KINDS {
            let mut e = Expansion::default();
            let default = e.is_expanded("x", kind);

            assert!(e.set("x", kind, !default), "{kind:?}: flip is a change");
            assert_eq!(e.is_expanded("x", kind), !default);
            assert!(e.set("x", kind, default), "{kind:?}: restore is a change");
            assert_eq!(e.is_expanded("x", kind), default);
            assert!(e.0.iter().all(HashSet::is_empty), "{kind:?}: a default is never stored");
        }
    }

    #[test]
    fn setting_the_current_state_is_not_a_change() {
        for kind in KINDS {
            let mut e = Expansion::default();
            let default = e.is_expanded("x", kind);
            assert!(!e.set("x", kind, default), "{kind:?}: already at default");

            e.set("x", kind, !default);
            assert!(!e.set("x", kind, !default), "{kind:?}: already flipped");
        }
    }

    #[test]
    fn kinds_do_not_share_state() {
        let mut e = Expansion::default();
        e.set("same-id", RowKind::File, true);
        assert!(e.is_expanded("same-id", RowKind::Symbol), "symbol still at its default");
    }
}
