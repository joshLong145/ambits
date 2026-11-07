//! Spreading independent work across cores.

use std::sync::atomic::{AtomicUsize, Ordering};

/// `work` over each of `items` on every core, the results in item order.
///
/// Workers take the next item from a shared counter rather than a fixed share
/// of the list: file sizes vary by orders of magnitude, and on a machine with
/// performance and efficiency cores a fixed share leaves the fast ones idle
/// while the slowest finishes.
///
/// A worker that panics loses only the results it was holding, which is the
/// caller's whole answer minus those items, not a failed command.
pub fn par_map<'a, T: Sync, R: Send>(items: &'a [T], work: impl Fn(&'a T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(items.len());
    let next = AtomicUsize::new(0);

    let mut done: Vec<(usize, R)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let at = next.fetch_add(1, Ordering::Relaxed);
                        let Some(item) = items.get(at) else {
                            return done;
                        };
                        done.push((at, work(item)));
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .filter_map(|h| h.join().ok())
            .flatten()
            .collect()
    });

    done.sort_unstable_by_key(|(at, _)| *at);
    done.into_iter().map(|(_, result)| result).collect()
}

#[cfg(test)]
mod tests {
    use super::par_map;

    #[test]
    fn results_keep_the_order_of_their_items() {
        let items: Vec<u32> = (0..1000).collect();
        let doubled = par_map(&items, |n| n * 2);
        assert_eq!(doubled, items.iter().map(|n| n * 2).collect::<Vec<_>>());
    }

    #[test]
    fn nothing_to_do_is_nothing_done() {
        assert!(par_map(&[] as &[u32], |n| *n).is_empty());
    }
}
