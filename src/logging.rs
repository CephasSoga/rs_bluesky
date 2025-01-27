use tracing::{span, info, debug, error, warn, trace};
use tracing_subscriber;
use tracing_subscriber::FmtSubscriber;

pub enum LogLevel {
    Trace, Info, Debug, Warn, Error
}
impl LogLevel {
    pub fn to_log_level(&self) -> tracing::Level {
        match self {
            LogLevel::Trace => tracing::Level::TRACE,
            LogLevel::Info => tracing::Level::INFO,
            LogLevel::Debug => tracing::Level::DEBUG,
            LogLevel::Warn => tracing::Level::WARN,
            LogLevel::Error => tracing::Level::ERROR,
        }
    }

    pub fn from_str(s: &str) -> Self{
        match s {
            "trace" => LogLevel::Trace,
            "info" => LogLevel::Info,
            "debug" => LogLevel::Debug,
            "warn" => LogLevel::Warn,
            "error" => LogLevel::Error,
            _ => LogLevel::Trace,
        }
    }
}
impl Default for LogLevel {
    fn default() -> Self {
        LogLevel::Trace
    }
}

const SPAN_NAME: &str = "News data";
pub struct Logger;

impl Logger {
    /// Initialize the logger
    pub fn init(level: LogLevel) {
        tracing_subscriber::fmt()
            .with_max_level(level.to_log_level()) // Set the maximum log level
            .init();
    }

    pub fn init_with_subscriber() {
        let subscriber = FmtSubscriber::builder().finish();
        tracing::subscriber::set_global_default(subscriber)
            .expect("Setting default subscriber failed.");
    }

    pub fn with_span<T>(f: impl FnOnce() -> T) -> T {
        let span = span!(tracing::Level::INFO, SPAN_NAME);
        let _guard = span.enter(); // Enter the span context
        f() // Execute the function within the span
    }

    /// Log a trace-level message
    pub fn trace(message: &str) {
        trace!(target: "custom_logger", "{}", message);
    }

    /// Log a debug-level message
    pub fn debug(message: &str) {
        debug!(target: "custom_logger", "{}", message);
    }

    /// Log an info-level message
    pub fn info(message: &str) {
        info!(target: "custom_logger", "{}", message);
    }

    /// Log a warn-level message
    pub fn warn(message: &str) {
        warn!(target: "custom_logger", "{}", message);
    }

    /// Log an error-level message
    pub fn error(message: &str) {
        error!(target: "custom_logger", "{}", message);
    }
}

pub fn setup_logger(level: &str) {
    match level {
        "error" => Logger::init(LogLevel::Error),
        "warn" => Logger::init(LogLevel::Warn),
        "info" => Logger::init(LogLevel::Info),
        "debug" => Logger::init(LogLevel::Debug),
        "trace" => Logger::init(LogLevel::Trace),
        _ => Logger::init(LogLevel::Trace),
    }
}

pub fn test_() {
    // Initialize the logger
    Logger::init(LogLevel::Trace);

    // Example logs
    Logger::trace("This is a trace message.");
    Logger::debug("This is a debug message.");
    Logger::info("This is an info message.");
    Logger::warn("This is a warning message.");
    Logger::error("This is an error message.");
}
