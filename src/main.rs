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

use std::sync::Arc;
use anyhow::Result;


use crate::classifiers::naive_bayes::NaiveBayes;
use crate::jetstream::WebSocketProxyServer;

const VOCAB_PATH: &str = "C:\\Users\\cepha\\OneDrive\\Bureau\\Cube\\gaello v.2\\rs_bluesky\\vocab.json";
const W_PATH: &str = "C:\\Users\\cepha\\OneDrive\\Bureau\\Cube\\gaello v.2\\rs_bluesky\\w.npy";
const NB_PATH: &str = "C:\\Users\\cepha\\OneDrive\\Bureau\\Cube\\gaello v.2\\rs_bluesky\\naive_bayes_model.json";

// -----------------------------------------------------------------------------
// Example main — run server at 127.0.0.1:9002
// -----------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing subscriber for logs
    tracing_subscriber::fmt::init();

    // Instantiate your NaiveBayes model (your file already provides constructors / load functions).
    // TODO: if your NaiveBayes has a loader (e.g. NaiveBayes::load(path)) use it here.
    let mut nb_model = NaiveBayes::load_from_file(NB_PATH)?;
    nb_model.build_dense_matrix();
    let nb = Arc::new(nb_model);

    // Create server
    let server = WebSocketProxyServer::new("0.0.0.0:9002", nb);

    // Optionally preload vocab and W if you have them at known paths:
    server.load_shared_artifacts(Some(VOCAB_PATH), Some(W_PATH));

    // Run server (this never returns unless error)
    server.run().await?;

    Ok(())
}

