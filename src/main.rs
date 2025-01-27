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

use std::sync::Arc;

use crate::classifiers::naive_bayes::NaiveBayes;

//use crate::classifiers::naive_bayes::example;
use crate::jetstream::ProxyServer;

#[tokio::main]
// *** Example usage (simplified) ***
pub async fn main() -> Result<(), Box<dyn std::error::Error>>{
    let classifier = Arc::new(NaiveBayes::load_from_file("naive_bayes_model.json")?);
    let address = "0.0.0.0:8080";
    let mut ws_server = ProxyServer::new(address, classifier);
    ws_server.run().await?;
    Ok(())
}