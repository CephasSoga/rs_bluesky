use std::fmt;
use std::hash::Hash;

use serde::Deserialize;
use config::{builder::DefaultState, ConfigBuilder, ConfigError, File};


#[derive(Debug, Clone, Hash, Deserialize)]
pub struct ConfigHeader {
    msg: String,
}

#[derive(Debug, Clone, Hash, Deserialize)]
pub struct ApiConfig {
    pub token: String,
}

#[derive(Debug, Clone, Hash, Deserialize)]
pub struct WebsocketConfig {
    pub server_addr: String,
    pub buffer_size: usize,
}

#[derive(Clone, Hash, Debug, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
}

#[derive(Debug, Clone, Hash, Deserialize)]
pub struct Config {
    pub header: ConfigHeader,
    pub api: ApiConfig,
    pub websocket: WebsocketConfig,
}
impl Config {
    pub fn new() -> Result<Self, ConfigError> {
    // Builder
    let mut builder: ConfigBuilder<DefaultState> = ConfigBuilder::default(); // Use default() instead of new()

    // Start off by merging in the "default" configuration file
    builder = builder.add_source(File::with_name("config")); // Example of adding a file source


    // Build the configuration
    let config = builder.build()
        .map_err(|e| {
            return ConfigError::FileParse { uri: Some(e.to_string()), cause: Box::new(e) }
        })?;

    // Deserialize the configuration into our Config struct
    // return it
    config.try_deserialize()

    }
}

impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Format the fields of ValueConfig as needed
        write!(f, "{}", self.header.msg)
    }
}
