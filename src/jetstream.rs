#[allow(dead_code)]
#[allow(unused_imports)]

use async_tungstenite::tokio::{TokioAdapter, connect_async};
//use async_tungstenite::tungstenite::Message;
use futures_util::{stream::SplitStream, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::{self, Sender};
use tokio::task;
use tokio_stream::wrappers::ReceiverStream;
//use futures_util::stream::SplitSink;
use tokio::net::TcpStream;
use async_tungstenite::WebSocketStream;
use tokio_native_tls::TlsStream;
use async_tungstenite::stream::Stream;
//use time::OffsetDateTime;

use crate::classifiers::naive_bayes::NaiveBayes;
use crate::config::Config;


type WebSocketStreamType = WebSocketStream<Stream<TokioAdapter<TcpStream>, TokioAdapter<TlsStream<TcpStream>>>>;


#[derive(Deserialize)]
struct Event {
    time_us: i128,
    did: String,
    commit: Option<Commit>,
}


#[derive(Deserialize)]
struct Commit {
    operation: String,
    collection: String,
    record: Option<serde_json::Value>,
}

#[derive(Deserialize, Serialize, Debug)]
struct FeedPost {
    text: String,
}

/// A struct to handle WebSocket streams
struct JetStream {
    receiver: SplitStream<WebSocketStreamType>,
    sender: Sender<String>,
}

impl JetStream {
    /// Create a new JetStream instance
    pub async fn new(sender: Sender<String>) -> Self {
        let config = Config::new().unwrap();

        let ws_addr = config.websocket.server_addr;

        let (ws_stream, _) = connect_async(ws_addr).await.expect("Failed to connect");
        let (_, read) = ws_stream.split();

        JetStream {
            receiver: read,
            sender,
        }
    }

    /// Process incoming WebSocket messages
    pub async fn process_messages(&mut self, classifier: &NaiveBayes) {
        while let Some(msg) = self.receiver.next().await {
            match msg {
                Ok(msg) if msg.is_text() => {
                    let text = msg.into_text().unwrap();
                    match serde_json::from_str::<Event>(&text) {
                        Ok(event) => {
                            let sender_clone = self.sender.clone();
                            let classifier_clone = classifier.clone();
                            task::spawn(async move {
                                JetStream::handle_event(event, sender_clone, &classifier_clone).await;
                            });
                        }
                        Err(err) => {
                            eprintln!("Failed to parse event: {}", err);
                        }
                    }
                }
                Ok(_) => {}
                Err(err) => {
                    eprintln!("WebSocket error: {}", err);
                }
            }
        }

        println!("Disconnected from Jetstream.");
    }

    /// Handle individual events
    async fn handle_event(event: Event, sender: Sender<String>, classifier: &NaiveBayes) {
        if let Some(commit) = event.commit {
            if commit.operation == "create" || commit.operation == "update" {
                if commit.collection == "app.bsky.feed.post" {
                    if let Some(record) = commit.record {
                        if let Ok(post) = serde_json::from_value::<FeedPost>(record) {
                            if let Some(label) = classifier.classify(&post.text) {
                                if label == "economy" || label == "stock_market" || label == "trading" {
                                    let log = format!(
                                        "{} |({})| {}",
                                        time::OffsetDateTime::from_unix_timestamp_nanos(event.time_us * 1000)
                                            .unwrap()
                                            .microsecond(),
                                        event.did,
                                        post.text
                                    );
                                    sender.send(log).await.unwrap();
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


//#[tokio::main]
pub async fn example() {
    let (sender, receiver) = mpsc::channel::<String>(100);
    let receiver_stream = ReceiverStream::new(receiver);

    // Spawn a task to log filtered posts
    task::spawn(async move {
        receiver_stream.for_each(|log| {
            println!("{}", log);
            futures_util::future::ready(())
        })
        .await;
    });

    // Create and train the Naive Bayes classifier
    let classifier = NaiveBayes::load_from_file("naive_bayes_model.json").unwrap();

    // Connect to Jetstream and process messages
    let mut jetstream = JetStream::new(sender).await;
    jetstream.process_messages(&classifier).await;
}
