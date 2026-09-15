//! Minimal stderr logger; avoids a dependency for four lines of formatting.

use log::{Level, LevelFilter, Log, Metadata, Record};

struct StderrLogger;

impl Log for StderrLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.level() <= Level::Info || metadata.target().starts_with("mc2")
    }

    fn log(&self, record: &Record<'_>) {
        if self.enabled(record.metadata()) && !record.target().starts_with("wgpu_hal") {
            eprintln!(
                "[{:<5} {}] {}",
                record.level(),
                record.target(),
                record.args()
            );
        }
    }

    fn flush(&self) {}
}

pub fn init() {
    static LOGGER: StderrLogger = StderrLogger;
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(LevelFilter::Info);
    }
}
