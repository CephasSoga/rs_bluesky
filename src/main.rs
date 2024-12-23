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

use std::str::FromStr;

//use crate::classifiers::naive_bayes::example;
use crate::jetstream::example;

#[tokio::main]
async fn main() {

    let _commit = generic_types::Commit{
        operation: "sub".to_string(), 
        collection: "all".to_string(), 
        record: Some(serde_json::Value::from_str("{\"text\": \"Hello\"}").unwrap()),   
    };
    //example(commit);
    example().await;
}
