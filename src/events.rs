use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

use crossterm::event::{self, Event, KeyEvent, MouseEvent};

use ambits::ingest::AgentToolCall;

/// Unified application event.
#[derive(Debug)]
pub enum AppEvent {
    Key(KeyEvent),
    Mouse(MouseEvent),
    FileChanged(PathBuf),
    AgentEvent(AgentToolCall),
    Tick,
}

/// Spawn a thread that polls crossterm key events and sends them to the channel.
pub fn spawn_key_reader(tx: mpsc::Sender<AppEvent>) {
    std::thread::spawn(move || loop {
        if event::poll(Duration::from_millis(50)).unwrap_or(false) {
            match event::read() {
                Ok(Event::Key(key)) => {
                    if tx.send(AppEvent::Key(key)).is_err() {
                        break;
                    }
                }
                Ok(Event::Mouse(mouse)) => {
                    if tx.send(AppEvent::Mouse(mouse)).is_err() {
                        break;
                    }
                }
                _ => {}
            }
        }
    });
}

/// Spawn a tick timer that sends Tick events at the given interval.
pub fn spawn_tick_timer(tx: mpsc::Sender<AppEvent>, interval: Duration) {
    std::thread::spawn(move || loop {
        std::thread::sleep(interval);
        if tx.send(AppEvent::Tick).is_err() {
            break;
        }
    });
}
