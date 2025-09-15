//pub mod test2;
//pub mod test;
//pub mod test3;
//pub mod test4;
//pub mod test5;
pub mod  tls;
pub mod config;
pub mod jetstream;
pub mod classifiers;
pub mod generic_types;
pub mod logging;
pub mod utils;
pub mod request_parser;
pub mod options;

use anyhow::{Ok, Result};

use crate::jetstream::exec;

// -----------------------------------------------------------------------------
// Example main — run server at 127.0.0.1:9002
// -----------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let _ = exec("0.0.0.0:9002").await?;
    Ok(())
}

