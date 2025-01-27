#![allow(dead_code)]
#![allow(unused_imports)]

use std::sync::Arc;

//use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tokio::sync::mpsc::{Sender, Receiver};
use serde_json::{to_value, to_string, Value};
use tracing::{info, error, warn, debug};
use async_tungstenite::tokio::{TokioAdapter, connect_async};
//use async_tungstenite::tungstenite::Message;
use futures_util::{stream::SplitStream, StreamExt, SinkExt, Future};
use serde::{Deserialize, Serialize};
use tokio::task;
use tokio_stream::wrappers::ReceiverStream;
use async_tungstenite::tungstenite::protocol::Message;
use async_tungstenite::tungstenite::error::Error;
//use futures_util::stream::SplitSink;
use tokio::net::{TcpListener, TcpStream, lookup_host};
use async_tungstenite::WebSocketStream;
use tokio_native_tls::TlsStream;
use async_tungstenite::stream::Stream;
//use time::OffsetDateTime;
use async_tungstenite::tokio::accept_async_with_config;
use tungstenite::protocol::WebSocketConfig;

use crate::classifiers::naive_bayes::NaiveBayes;
use crate::config::Config;
use crate::logging::setup_logger;


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

#[derive(Deserialize, Serialize, Debug)]
struct Log {
    text: String,
    time: i128,
    did: String,
}

/// A struct to handle WebSocket streams
pub struct JetStream {
    receiver: SplitStream<WebSocketStreamType>,
    sender: Sender<String>,
    close_tx: Sender<()>, // Add a close sender
}

impl JetStream {
    /// Create a new JetStream instance
    pub async fn new(sender: Sender<String>, close_tx: Sender<()>) -> Self {
        let config = Config::new().unwrap();

        let ws_addr = config.websocket.server_addr;

        let (ws_stream, _) = connect_async(ws_addr).await.expect("Failed to connect");
        let (_, read) = ws_stream.split();

        JetStream {
            receiver: read,
            sender,
            close_tx,
        }
    }

    /// Process incoming WebSocket messages
    pub async fn handle(&mut self, classifier: &Arc<NaiveBayes>) {
        while let Some(msg) = self.receiver.next().await {
            match msg {
                Ok(msg) if msg.is_text() => {
                    let text = msg.into_text().unwrap();
                    match serde_json::from_str::<Event>(&text) {
                        Ok(event) => {
                            let sender_clone = self.sender.clone();
                            let classifier_clone = classifier.clone();
                            task::spawn(async move {
                                JetStream::forward(event, sender_clone, &classifier_clone).await;
                            });
                        }
                        Err(err) => {
                            error!("Failed to parse event. | Error: {}", err);
                        }
                    }
                }
                Ok(_) => {}
                Err(err) => {
                    error!("JetStream handle error. | Error:  {}", err);
                }
            }
        }
        info!("Disconnected from Jetstream.");
        // Signal closure
        let _ = self.close_tx.send(()).await; // Send close signal (ignore error)
    }

    /// Handle individual events
    async fn forward(event: Event, sender: Sender<String>, classifier: &Arc<NaiveBayes>) {
        if let Some(commit) = event.commit {
            if commit.operation == "create" || commit.operation == "update" {
                if commit.collection == "app.bsky.feed.post" {
                    if let Some(record) = commit.record {
                        if let Ok(post) = serde_json::from_value::<FeedPost>(record) {
                            if let Some(label) = classifier.classify(&post.text) {
                                if label == "economy" || label == "stock_market" || label == "trading" {
                                    let log = Log {
                                        text: post.text,
                                        time: time::OffsetDateTime::from_unix_timestamp_nanos(event.time_us * 1000)
                                            .unwrap()
                                            .microsecond() as i128,
                                        did: event
                                        .did,
                                    };
                                    let log_repr = to_string(&log)
                                        .inspect(|s| info!("Fowarded some log.| Log: {}", s))
                                        .map_err(|err| warn!("Some error during log fowarding. | Error: {}", err))
                                        .expect("Failed to Json encode Log ");
                                    sender.send(log_repr).await.unwrap();
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

pub struct  ProxyServer{
    address: String,
    classifier: Arc<NaiveBayes>,
}
impl ProxyServer {
    pub fn new(addr: &str, classifier: Arc<NaiveBayes>) -> Self {
        setup_logger("trace");
        Self {
            address: addr.to_string(),
            classifier,
        }
    }

    fn process_message(json: Value) -> Option<String> {
        if let Ok(post) = serde_json::from_value::<FeedPost>(json) {
            Some(format!("Processed message: {}", post.text))
        } else {
            None
        }
    }

    async fn handle_connection(stream: TcpStream, classifier: Arc<NaiveBayes>) {
        let config = Some(WebSocketConfig::default());

        let ws_stream = match accept_async_with_config(stream, config).await {
            Ok(ws_stream) => ws_stream,
            Err(e) => {
                error!("Error during handshake: {}", e);
                return;
            }
        };

        let (mut write, mut read) = ws_stream.split();
        let (tx_ws, rx_ws) = mpsc::channel::<String>(100); // Channel for client-bound messages
        let (tx_js, rx_js) = mpsc::channel::<String>(100); // Channel for JS-bound messages
        let (close_tx, mut close_rx) = mpsc::channel::<()>(1); // Channel to signal closure
        let tx_ws = tx_ws.clone();

        // *** Create and launch JetStream for this connection ***
        let mut jetstream = JetStream::new(tx_js, close_tx).await;
        let classifier_clone_js = classifier.clone();
        let js_handle  = tokio::spawn(async move {
            jetstream.handle(&classifier_clone_js).await;
        });

        // Spawn task to handle outgoing messages to client
        let ws_write_handle = tokio::spawn(async move {
            let mut receiver_stream = ReceiverStream::new(rx_ws); // Convert Receiver to a stream
            while let Some(msg) = receiver_stream.next().await { // Use the stream's next method
                if write.send(Message::Text(msg)).await.is_err() {
                    break;
                }
            }
        });

        // *** Forward messages from JetStream to client ***
        let js_to_ws_handle  = {
            let tx_ws_ = tx_ws.clone();
            tokio::spawn(async move {
                let tx_ws = tx_ws_.clone();
                let mut receiver_stream = ReceiverStream::new(rx_js);
                while let Some(msg) = receiver_stream.next().await {
                    if tx_ws.send(msg).await.is_err() {
                        break;
                    }
                }
            })
        };

        let tx_ws_ = tx_ws.clone();
        // Handle incoming messages
        while let Some(msg) = read.next().await {
            let tx_ws = tx_ws_.clone();
            match msg {
                Ok(Message::Text(text)) => {
                    match serde_json::from_str::<Value>(&text) {
                        Ok(json) => {
                            //Spawn a task to send filtered posts
                            if let Some(log) = Self::process_message(json) {
                                if tx_ws.send(log).await.is_err() {
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            error!("Failed to parse JSON: {}", e);
                            if let Err(_) = tx_ws.send("Invalid JSON".to_string()).await {
                                break;
                            }
                        }
                    }
                }
                Ok(Message::Close(_)) => break,
                Err(e) => {
                    warn!("Error receiving message: {}", e);
                    break;
                }
                _ => {}
            }
        }

        tokio::select! {
            _ = js_handle => {
                info!("Client disconnected (JS write).");
            }
            _ = ws_write_handle => {
                info!("Client disconnected (WS write).");
            }
            _ = js_to_ws_handle => {
                info!("Client disconnected (JS to WS forward).");
            }
            _ = close_rx.recv() => { // Wait for JetStream closure signal
                info!("JetStream disconnected.");
            }
        }
        // Abort all tasks to ensure proper cleanup
        //js_handle.abort();
        //ws_write_handle.abort();
        //js_to_ws_handle.abort();

        info!("Connection handler finished.");

    }

    pub async fn run(&mut self) -> Result<(), Error> {
        info!(message="Resolving address", addr=&(*self.address));
        let mut addrs = lookup_host(&(*self.address)).await
            .map_err(|e| error!("Error resolving address: {}", e.to_string()))
            .unwrap();

        let addr = addrs.next()
            .ok_or_else(|| {
            error!(err="Failed to resolve address", addr=&(*self.address));
            Error::Url(tungstenite::error::UrlError::NoHostName)
        })?;

        info!("Setting address: {}", self.address);
        let listener = TcpListener::bind(&addr)
            .await
            .map_err(|e| error!("Error: {}", e.to_string()))
            .unwrap();

        info!("WebSocket server listening on: {}", self.address);
        while let Ok((stream, _addr)) = listener.accept().await {
            let classifier_ = self.classifier.clone();
            tokio::spawn(async move {
                ProxyServer::handle_connection(stream, classifier_).await;
            });
        }

        Ok(())
    }
    
}

