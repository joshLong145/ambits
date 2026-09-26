//! Command output on stdout, and what a closed pipe means for it.
//!
//! `ambits … | head` is ordinary use: the reader exits once it has what it
//! wants, and the next write fails with `BrokenPipe`. That is the consumer
//! ending the conversation, not a failure of ours — ripgrep exits 0 in the
//! same spot, and so does ambits. Rust ignores SIGPIPE, so left alone a
//! `println!` panics (exit 101) and a propagated write error looks like any
//! other error (exit 2 for the search commands).
//!
//! The convention: commands write through a locked stdout with
//! `writeln!(out, …)?` — never `println!`, which panics — and let the error
//! propagate. `main` then decides once, with [`is_broken_pipe`], instead of
//! every command deciding for itself. The check cannot tell *which* pipe
//! broke; that is sound only while stdout is the one pipe ambits writes to
//! (the `hostname` probe is read, and the editor inherits the terminal). A
//! future writer to a child's stdin or a socket must handle its own
//! `BrokenPipe` before the error reaches `main`.
//!
//! Diagnostics go to stderr through [`try_eprintln!`](crate::try_eprintln),
//! never `eprintln!`: with stderr closed, `eprintln!` panics, the panic hook
//! then fails writing its own report to the same stream, and the process
//! aborts.

use std::io;

/// `eprintln!` that cannot panic. A diagnostic that cannot be delivered has no
/// one left to deliver it to, so the write error is dropped.
#[macro_export]
macro_rules! try_eprintln {
    ($($arg:tt)*) => {
        $crate::output::write_stderr_line(format_args!($($arg)*))
    };
}

#[doc(hidden)]
pub fn write_stderr_line(args: std::fmt::Arguments<'_>) {
    use io::Write;
    let _ = writeln!(io::stderr().lock(), "{args}");
}

/// Whether `err` was caused, anywhere in its chain, by writing to a pipe
/// whose reader has gone away.
pub fn is_broken_pipe(err: &color_eyre::Report) -> bool {
    err.chain().any(|cause| {
        cause
            .downcast_ref::<io::Error>()
            .is_some_and(|e| e.kind() == io::ErrorKind::BrokenPipe)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use color_eyre::eyre::WrapErr;

    fn io_report(kind: io::ErrorKind) -> color_eyre::Report {
        color_eyre::Report::new(io::Error::from(kind))
    }

    #[test]
    fn a_broken_pipe_is_recognised() {
        assert!(is_broken_pipe(&io_report(io::ErrorKind::BrokenPipe)));
    }

    #[test]
    fn a_broken_pipe_under_added_context_is_still_recognised() {
        let err: Result<(), _> = Err(io::Error::from(io::ErrorKind::BrokenPipe));
        let wrapped = err.wrap_err("writing results").unwrap_err();
        assert!(is_broken_pipe(&wrapped));
    }

    #[test]
    fn other_errors_are_not() {
        assert!(!is_broken_pipe(&io_report(io::ErrorKind::PermissionDenied)));
        assert!(!is_broken_pipe(&color_eyre::eyre::eyre!("no such symbol")));
    }
}
