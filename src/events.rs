use std::path::PathBuf;
use std::time::Duration;

use crossterm::event::{self, Event, KeyEvent, MouseEvent};

/// Something the TUI loop must react to.
///
/// Only what a producer actually sends. Session activity is deliberately not
/// here: agent events, clears and compactions reach the app through
/// `TuiSession::handle_tick`, which polls the log tailer, rather than being
/// pushed down this channel. Variants existed for all three and were matched
/// in the loop, but nothing had constructed them since that responsibility
/// moved — three arms that could never fire, implying a data path that no
/// longer exists.
#[derive(Debug)]
pub enum AppEvent {
    Key(KeyEvent),
    Mouse(MouseEvent),
    FileChanged(PathBuf),
    Tick,
}

/// Spawn a thread that polls crossterm key events and sends them to the channel.
pub fn spawn_key_reader(tx: flume::Sender<AppEvent>) {
    std::thread::spawn(move || loop {
        if !event::poll(Duration::from_millis(50)).unwrap_or(false) {
            continue;
        }
        // Translate first, send once. A failed send means the receiver is gone
        // and the thread should stop; that is one fact about the channel, not
        // one per event kind, and duplicating the check per arm is what made
        // this read as two unrelated branches.
        let event = match event::read() {
            Ok(Event::Key(key)) => AppEvent::Key(key),
            Ok(Event::Mouse(mouse)) => AppEvent::Mouse(mouse),
            _ => continue,
        };
        if tx.send(event).is_err() {
            break;
        }
    });
}

/// Spawn a tick timer that sends Tick events at the given interval.
pub fn spawn_tick_timer(tx: flume::Sender<AppEvent>, interval: Duration) {
    std::thread::spawn(move || loop {
        std::thread::sleep(interval);
        if tx.send(AppEvent::Tick).is_err() {
            break;
        }
    });
}
