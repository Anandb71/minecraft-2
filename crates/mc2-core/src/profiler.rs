//! CPU scope profiler feeding the on-screen HUD.
//!
//! Scopes from any thread accumulate into a shared per-frame table. At the end
//! of each frame the totals move into [`RollingStats`] rows, so a system that
//! runs twice in a frame reports its summed cost and a system that skipped a
//! frame reports zero, which keeps p99 honest for amortised work. The same
//! scopes are forwarded to puffin for flame graph captures.

use crate::stats::RollingStats;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

pub use puffin;

#[derive(Default)]
struct Table {
    /// Accumulated milliseconds per scope for the frame in progress.
    pending: Vec<(&'static str, f32)>,
    rows: Vec<ProfileRow>,
}

#[derive(Clone, Debug)]
pub struct ProfileRow {
    pub name: &'static str,
    pub stats: RollingStats,
}

fn table() -> &'static Mutex<Table> {
    static TABLE: OnceLock<Mutex<Table>> = OnceLock::new();
    TABLE.get_or_init(|| Mutex::new(Table::default()))
}

/// Adds `ms` to the named scope for the current frame.
pub fn record(name: &'static str, ms: f32) {
    let mut t = table().lock().unwrap_or_else(|e| e.into_inner());
    match t.pending.iter_mut().find(|(n, _)| *n == name) {
        Some((_, acc)) => *acc += ms,
        None => t.pending.push((name, ms)),
    }
}

/// Closes the frame: pending totals become samples, absent rows record zero.
pub fn end_frame() {
    let mut guard = table().lock().unwrap_or_else(|e| e.into_inner());
    let t = &mut *guard;
    for row in &mut t.rows {
        let ms = t
            .pending
            .iter()
            .find(|(n, _)| *n == row.name)
            .map_or(0.0, |(_, ms)| *ms);
        row.stats.push(ms);
    }
    for (name, ms) in t.pending.drain(..) {
        if !t.rows.iter().any(|r| r.name == name) {
            let mut stats = RollingStats::default();
            stats.push(ms);
            t.rows.push(ProfileRow { name, stats });
        }
    }
    drop(guard);
    puffin::GlobalProfiler::lock().new_frame();
}

/// Snapshot of every row, in first-seen order.
pub fn rows() -> Vec<ProfileRow> {
    table()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .rows
        .clone()
}

/// RAII timer; prefer the [`scope!`](crate::scope) macro.
pub struct ScopeGuard {
    name: &'static str,
    start: Instant,
}

impl ScopeGuard {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            start: Instant::now(),
        }
    }
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        record(self.name, self.start.elapsed().as_secs_f32() * 1000.0);
    }
}

/// Times the rest of the enclosing block under `name`.
#[macro_export]
macro_rules! scope {
    ($name:expr) => {
        let _mc2_scope_guard = $crate::profiler::ScopeGuard::new($name);
        $crate::profiler::puffin::profile_scope!($name);
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    // The table is process global; serialise tests that close frames.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn scopes_accumulate_and_absent_rows_record_zero() {
        let _serial = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        record("test.a", 1.0);
        record("test.a", 2.0);
        end_frame();
        let row = rows().into_iter().find(|r| r.name == "test.a").unwrap();
        assert!((row.stats.last() - 3.0).abs() < 1e-6);
        end_frame();
        let row = rows().into_iter().find(|r| r.name == "test.a").unwrap();
        assert_eq!(row.stats.last(), 0.0);
    }

    #[test]
    fn guard_records_elapsed_time() {
        let _serial = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        {
            crate::scope!("test.guard");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        end_frame();
        let row = rows().into_iter().find(|r| r.name == "test.guard").unwrap();
        assert!(row.stats.max() >= 1.5);
    }
}
