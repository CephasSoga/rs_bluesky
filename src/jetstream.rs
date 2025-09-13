

// batch_nb_complete.rs
//
// Single-file integration:
// - WebSocket proxy server (accepts client connections)
// - JetStream (ingests external events — we reuse your existing event shape)
// - BatchManager (size + timeout flush)
// - NdarrayTransformer (text -> Array2 bag-of-words)
// - Optional vectorized classification using W.npy (Array2) with ndarray
// - Fallback per-message classification using your provided NaiveBayes
// - Pool workers run batch work in blocking threads and forward results to WS clients
//
// Lots of comments and TODOs throughout so you can revisit pieces later.
//
// NOTES:
// - This file assumes your NaiveBayes implementation is reachable as
//   crate::naive_bayes::NaiveBayes (adjust the `use` line if needed).
// - Before running: add dependencies to Cargo.toml (see top of message).
// - To enable vectorized path: prepare `vocab.json` (Vec<String>) and `W.npy` (num_classes x vocab_size).
//   If either is missing, the code will fall back to per-message NaiveBayes classification.
//
// Author: ChatGPT (integrated with your NaiveBayes)
//

#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(clippy::needless_pass_by_value)]

use std::sync::Arc;
use std::collections::HashMap;
use std::path::Path;

use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tokio::sync::mpsc::{Sender, Receiver};
use tokio::task;
use tokio_stream::wrappers::ReceiverStream;
use futures_util::{StreamExt, SinkExt};
use async_tungstenite::tungstenite::protocol::Message;
use async_tungstenite::tungstenite::error::Error;
use async_tungstenite::tokio::{TokioAdapter, connect_async};
use async_tungstenite::tokio::accept_async_with_config;
use tungstenite::protocol::WebSocketConfig;
use tokio::net::{TcpListener, TcpStream, lookup_host};
use tokio::sync::mpsc::error::TrySendError;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{info, warn, error, debug};

use ndarray::{Array2, ArrayView2, Axis};
use ndarray_npy::read_npy;
use once_cell::sync::OnceCell;
use anyhow::{Result, Context};

const LABELS_PATH: &str = "C:\\Users\\cepha\\OneDrive\\Bureau\\Cube\\gaello v.2\\rs_bluesky\\labels.txt";
const VOCAB_PATH: &str = "C:\\Users\\cepha\\OneDrive\\Bureau\\Cube\\gaello v.2\\rs_bluesky\\vocab.json";
const W_PATH: &str = "C:\\Users\\cepha\\OneDrive\\Bureau\\Cube\\gaello v.2\\rs_bluesky\\w.npy";

// -----------------------------------------------------------------------------
// Replace or adapt this import to match where your naive_bayes.rs module lives.
// In the original repo you used: crate::classifiers::naive_bayes::NaiveBayes
// Here we assume it's at crate::naive_bayes::NaiveBayes (change if different).
// -----------------------------------------------------------------------------
//use crate::naive_bayes::NaiveBayes; // <- adjust path if needed
use crate::classifiers::naive_bayes::NaiveBayes;
use crate::config::Config;
use crate::logging::setup_logger;
use crate::request_parser::params::TaskFunction;
use crate::request_parser::parser::CallParser;

// -----------------------------------------------------------------------------
// Domain types
// -----------------------------------------------------------------------------

/// Simple message item we transport through pipeline
#[derive(Clone, Debug)]
pub struct MessageItem {
    pub id: usize,
    pub text: String,
}

/// BatchMatrix: holds the ndarray matrix and original MessageItems so we can map results back
#[derive(Clone)]
pub struct BatchMatrix {
    pub x: Array2<f32>,          // shape: (batch_size, vocab_size)
    pub items: Vec<MessageItem>, // length == batch_size
}

// -----------------------------------------------------------------------------
// Tokenizer, Vocab, Transformer
// -----------------------------------------------------------------------------

/// Simple Vocab container mapping token -> index
#[derive(Clone)]
pub struct Vocab {
    pub token_to_idx: HashMap<String, usize>,
}

impl Vocab {
    /// Build from an iterator of tokens (ordered) - tokens[0] -> idx 0, etc.
    pub fn new_from_iter<I: IntoIterator<Item = String>>(iter: I) -> Self {
        let mut token_to_idx = HashMap::new();
        for (i, tok) in iter.into_iter().enumerate() {
            token_to_idx.insert(tok, i);
        }
        Self { token_to_idx }
    }

    /// Load vocab from a JSON file containing an array of tokens: ["the","to",...]
    pub fn load_from_json<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        let json: Value = serde_json::from_slice(&bytes)?;

        let tokens: Vec<String> = match json {
            Value::Array(arr) => arr
                .into_iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
            Value::Object(obj) => obj
                .keys()
                .cloned()
                .collect(),
            _ => anyhow::bail!("Unsupported vocab format"),
        };

        Ok(Vocab::new_from_iter(tokens))
    }

    pub fn len(&self) -> usize { self.token_to_idx.len() }
}


pub struct  Labels {
    pub vec: Vec<String>,
}
impl Labels {
    pub fn new_from_iter<I: IntoIterator<Item = String>>(iter: I) -> Self {
        Self { vec: iter.into_iter().collect() }
    }

    pub fn load_from_txt<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        // Read file as text, interpret one label per line, skip empty lines & trim whitespace.
        let s = std::fs::read_to_string(path)?;
        let mut vec = Vec::new();
        for line in s.lines() {
            let t = line.trim();
            if !t.is_empty() {
                vec.push(t.to_string());
            }
        }
        Ok(Labels { vec })
    }
    

    pub fn get(&self, idx: usize) -> Option<&String> {
        self.vec.get(idx)
    }

    pub fn len(&self) -> usize {
        self.vec.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vec.is_empty()
    }

}

impl IntoIterator for Labels {
    type Item = String;
    type IntoIter = std::vec::IntoIter<String>;

    fn into_iter(self) -> Self::IntoIter {
        self.vec.into_iter()
    }
}

impl std::fmt::Display for Labels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]", self.vec.join(", "))
    }
}



/// Tokenizer: simple ASCII/word tokenizer. Replace with your project's tokenizer for exact match.
pub struct Tokenizer;

impl Tokenizer {
    /// Basic word tokenizer: extracts word characters and lowercases.
    pub fn tokenize<'a>(s: &'a str) -> impl Iterator<Item = String> + 'a {
        s.split_whitespace()
            .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|t| !t.is_empty())
            .map(|t| t.to_string())
    }
}

/// Transformer trait — convert a batch of MessageItems into BatchMatrix
pub trait Transformer: Send + Sync {
    fn to_batch_matrix(&self, batch: &[MessageItem]) -> BatchMatrix;
}

/// NdarrayTransformer: real transformer building Array2 bag-of-words using provided Vocab.
pub struct NdarrayTransformer {
    pub vocab: Arc<Vocab>,
}

impl NdarrayTransformer {
    pub fn new(vocab: Arc<Vocab>) -> Self {
        Self { vocab }
    }
}

impl Transformer for NdarrayTransformer {
    fn to_batch_matrix(&self, batch: &[MessageItem]) -> BatchMatrix {
        let vocab_size = self.vocab.len();
        let batch_size = batch.len();

        // Create a matrix (batch_size, vocab_size)
        let mut x = Array2::<f32>::zeros((batch_size, vocab_size));

        for (i, msg) in batch.iter().enumerate() {
            for token in Tokenizer::tokenize(&msg.text) {
                if let Some(&idx) = self.vocab.token_to_idx.get(&token) {
                    // Count occurrences (bag-of-words)
                    x[[i, idx]] += 1.0;
                }
            }
        }

        BatchMatrix { x, items: batch.to_vec() }
    }
}

// -----------------------------------------------------------------------------
// Dispatcher (decides which pool gets the BatchMatrix)
// -----------------------------------------------------------------------------

/// Dispatcher: route a BatchMatrix to a pool worker
pub struct Dispatcher {
    // typed channels to pools: BatchMatrix channel per pool
    senders: Arc<Vec<mpsc::Sender<Arc<BatchMatrix>>>>,
    rr: std::sync::atomic::AtomicUsize,
}

impl Clone for Dispatcher {
    fn clone(&self) -> Self {
        Dispatcher {
            senders: Arc::clone(&self.senders),
            rr: std::sync::atomic::AtomicUsize::new(self.rr.load(std::sync::atomic::Ordering::Relaxed)),
        }
    }
}

impl Dispatcher {
    pub fn new(senders: Vec<mpsc::Sender<Arc<BatchMatrix>>>) -> Self {
        Self { senders: Arc::new(senders), rr: std::sync::atomic::AtomicUsize::new(0) }
    }

    /// Round-robin dispatch but non-blocking try_send; if the chosen pool is full we try next
    /// up to senders.len() times. If all full, we drop the batch (baseline). You can swap
    /// this policy for backpressure or persistent queue later.
    pub async fn dispatch(&self, batch: Arc<BatchMatrix>) {
        let n = self.senders.len();
        if n == 0 {
            warn!("Dispatcher has no senders; dropping batch");
            return;
        }
        let start = self.rr.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % n;
        for i in 0..n {
            let idx = (start + i) % n;
            let tx = &self.senders[idx];
            match tx.try_send(batch.clone()) {
                Ok(_) => return,
                Err(err) => match err {
                    // If send fails because channel is full, try next. If disconnected, log and continue.
                    tokio::sync::mpsc::error::TrySendError::Full(_) => continue,
                    tokio::sync::mpsc::error::TrySendError::Closed(_) => {
                        warn!("Dispatcher: pool {} channel closed", idx);
                        continue;
                    }  
                }
            }
        }
        // All queues full / failed - baseline policy: drop
        warn!("All pool queues full or failed; dropping batch (baseline)");
    }
}

// -----------------------------------------------------------------------------
// Pool worker: receives BatchMatrix and classifies them.
//
// Behavior:
// - If an Array2 W (num_classes x vocab_size) is available, compute scores = X.dot(&W.t())
//   and compute argmax per row (fast vectorized).
// - Else, fallback to calling NaiveBayes::classify(&text) for each item (via nb).
// - Results are sent over tx_results (if provided) which should be wired to WS writer or DB.
// -----------------------------------------------------------------------------

pub struct PoolWorker {
    id: usize,
    rx: mpsc::Receiver<Arc<BatchMatrix>>,
    // optional W matrix for vectorized classification (num_classes x vocab_size)
    w: Option<Arc<Array2<f32>>>,
    // fallback classifier
    nb: Option<Arc<NaiveBayes>>,
    // channel to send result JSON strings to writer
    tx_results: Option<mpsc::Sender<String>>,
    shared_labels: OnceCell<Arc<Labels>>,
}

impl PoolWorker {
    pub fn new(
        id: usize,
        rx: mpsc::Receiver<Arc<BatchMatrix>>,
        w: Option<Arc<Array2<f32>>>,
        nb: Option<Arc<NaiveBayes>>,
        tx_results: Option<mpsc::Sender<String>>,
        labels: Option<Arc<Labels>>,
    ) -> Self {
        let cell = OnceCell::new();
        if let Some(l) = labels {
            // We can unwrap safely because we're creating the OnceCell before any concurrent usage.
            let _ = cell.set(l);
        }
        Self { id, rx, w, nb, tx_results, shared_labels: cell }
    }

    /// Run worker loop; receives BatchMatrix and spawns blocking tasks for CPU-bound work.
    pub async fn run(mut self) {
        let labels = self.shared_labels.clone();
        while let Some(batch) = self.rx.recv().await {
            let batch_len = batch.items.len();
            let maybe_w = self.w.clone();
            let maybe_nb = self.nb.clone();
            let tx = self.tx_results.clone();
            let id = self.id;

            // Offload heavy computation to blocking threadpool
            let labels = labels.clone();
            task::spawn_blocking(move || {
                if let Some(wmat) = maybe_w {
                    // Vectorized path: scores = X.dot(W.t())
                    // X shape: (batch_size, vocab_size)
                    // W shape: (num_classes, vocab_size)
                    // W.t() shape: (vocab_size, num_classes)
                    // result shape: (batch_size, num_classes)
                    let scores: Array2<f32> = batch.x.dot(&wmat.t());

                    for (i, row) in scores.rows().into_iter().enumerate() {
                        // compute argmax manually (because ndarray ArgMax is limited)
                        let mut best_idx = 0usize;
                        let mut best_val = std::f32::NEG_INFINITY;
                        for (j, v) in row.iter().enumerate() {
                            if *v > best_val {
                                best_val = *v;
                                best_idx = j;
                            }
                        }
                        // Map result back to original message for context
                        let msg = &batch.items[i];

                        let label = labels
                            .get()
                            .and_then(|vec| vec.get(best_idx))
                            .cloned()
                            .unwrap_or_else(|| {
                                error!("Labels not set or index out of range");
                                "unknown".to_string()
                            });

                        // Build result payload — you may replace class_idx with actual label names if you have them
                        let payload = serde_json::json!({
                            "pool": id,
                            "id": msg.id,
                            "class_idx": best_idx,
                            "label": label,
                            "score": best_val,
                            "text": msg.text
                        });

                        //info!("[pool {}] msg {} -> class {} score {}", id, msg.id, best_idx, best_val);

                        if let Some(tx) = tx.as_ref() {
                            // best-effort async send: spawn a tiny task
                            let out = payload.to_string();
                            let tx = tx.clone();
                            let _ = tokio::spawn(async move {
                                let _ = tx.send(out).await;
                            });
                        }
                    }
                } else if let Some(nb) = maybe_nb {
                    // Collect all the texts from the batch into a Vec<String>
                    let texts: Vec<String> = batch.items.iter().map(|item| item.text.clone()).collect();
                    
                    // Perform a single, batched classification
                    let results = nb.classify_batch_dense(&texts);
                    
                    // Process the results and send them back
                    for (i, label) in results.into_iter().enumerate() {
                        let msg = &batch.items[i];
                        
                        //info!("[pool {}] msg {} -> label {}", id, msg.id, label);
                        
                        if let Some(tx) = tx.as_ref() {
                            let payload = serde_json::json!({
                                "pool": id,
                                "id": msg.id,
                                "label": label,
                                "text": msg.text
                            });
                            let out = payload.to_string();
                            let tx = tx.clone();
                            let _ = tokio::spawn(async move {
                                let _ = tx.send(out).await;
                            });
                        }
                    }
                } else {
                    // No classifier attached; just print
                    let batch = (*batch).clone();
                    for msg in batch.items.into_iter() {
                        //info!("[pool {}] msg {} - no classifier attached; text: {}", id, msg.id, msg.text);
                    }
                }
            });
            //info!("Pool {} done | Size {} ", id, batch_len);
        }
    }
}

// -----------------------------------------------------------------------------
// BatchManager: collects MessageItem into a buffer, flushes either when size hit,
// or when timeout expires. Exposes a `feed` method for incoming messages.
// -----------------------------------------------------------------------------

pub struct BatchManager {
    buffer: Vec<MessageItem>,
    max_size: usize,
    timeout_ms: u64,
    transformer: Arc<dyn Transformer>,
    dispatcher: Arc<Dispatcher>,
    // To avoid requiring a tokio handle for tick in sync contexts, we keep stateful flush semantics
    last_flush: std::time::Instant,
}

impl BatchManager {
    /// Create a new manager. timeout_ms = 0 disables timeout flushing.
    pub fn new(
        max_size: usize,
        timeout_ms: u64,
        transformer: Arc<dyn Transformer>,
        dispatcher: Arc<Dispatcher>,
    ) -> Self {
        Self {
            buffer: Vec::with_capacity(max_size),
            max_size,
            timeout_ms,
            transformer,
            dispatcher,
            last_flush: std::time::Instant::now(),
        }
    }

    /// Push a message into the buffer. If a batch is ready, it is dispatched asynchronously.
    pub async fn feed(&mut self, item: MessageItem) {
        self.buffer.push(item);
        let now = std::time::Instant::now();

        // If buffer full -> flush
        if self.buffer.len() >= self.max_size {
            self.flush().await;
            self.last_flush = now;
            return;
        }

        // Timeout-based flush
        if self.timeout_ms > 0 {
            let elapsed = now.duration_since(self.last_flush).as_millis() as u64;
            if elapsed >= self.timeout_ms && !self.buffer.is_empty() {
                self.flush().await;
                self.last_flush = now;
            }
        }
    }

    /// Force flush current buffer (if not empty)
    pub async fn flush(&mut self) {
        if self.buffer.is_empty() { return; }
        // Take ownership of current buffer
        let batch_items = std::mem::take(&mut self.buffer);
        // Transform into BatchMatrix
        let bm = self.transformer.to_batch_matrix(&batch_items);
        let bm = Arc::new(bm);
        // Dispatch (async)
        let disp = self.dispatcher.clone();
        disp.dispatch(bm).await;
    }
}


// -----------------------------------------------------------------------------
// JetStream: original upstream feed connection handler (adapted to push MessageItem to batch_tx)
//
// Note: for baseline we implement two use-cases:
//  - JetStream::handle_text(text) consumes event JSON strings and pushes eligible posts to batch_tx
//  - In earlier code you had JetStream connecting to an external address. If you still want that,
//    you can adapt handle_connect_async() to use connect_async and feed messages similarly.
// -----------------------------------------------------------------------------

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

#[derive(Deserialize, Serialize, Debug, Clone)]
struct FeedPost {
    text: String,
}

#[derive(Deserialize, Serialize, Debug)]
struct Log {
    text: String,
    time: i128,
    did: String,
}

#[derive(Deserialize, Serialize, Debug)]
struct ErrorLog {
    error: String,
}

pub struct JetStream {
    url: String,
    batch_tx: mpsc::Sender<MessageItem>,
}

impl JetStream {
    pub fn new(url: &str, batch_tx: mpsc::Sender<MessageItem>) -> Self {
        Self {
            url: url.to_string(),
            batch_tx,
        }
    }

    /// Connect to the Jetstream feed and continuously process events
    pub async fn run(&self) -> anyhow::Result<()> {
        let (ws_stream, _) = connect_async(&self.url).await?;
        let (_, mut read) = ws_stream.split();

        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(txt)) => {
                    self.handle_text(&txt).await;
                }
                Ok(Message::Close(_)) => {
                    info!("JetStream feed closed");
                    break;
                }
                Ok(_) => {} // ignore binary/ping/pong
                Err(e) => {
                    warn!("JetStream WS error: {}", e);
                    break;
                }
            }
        }

        Ok(())
    }

     /// Parse one raw JSON text message from Jetstream feed and forward eligible posts
    /// into the batch pipeline.
    pub async fn handle_text(&self, text: &str) {
        // Try to parse into your Event type
        let evt: Event = match serde_json::from_str(text) {
            Ok(e) => e,
            Err(e) => {
                warn!("JetStream: failed to parse Event JSON: {}", e);
                return;
            }
        };

        let commit = match evt.commit {
            Some(c) => c,
            None => return, // ignore events without commit
        };

        // Only forward create/update of feed posts
        if (commit.operation == "create" || commit.operation == "update")
            && commit.collection == "app.bsky.feed.post"
        {
            if let Some(record) = commit.record {
                // Try to decode into FeedPost struct
                match serde_json::from_value::<FeedPost>(record) {
                    Ok(post) => {
                        let msg = MessageItem {
                            id: rand::random::<usize>(),
                            text: post.text,
                        };
                        if let Err(e) = self.batch_tx.send(msg).await {
                            warn!("JetStream: failed to send MessageItem: {}", e);
                        }
                    }
                    Err(e) => {
                        warn!("JetStream: commit.record not a FeedPost: {}", e);
                    }
                }
            }
        }
    }
}


// -----------------------------------------------------------------------------
// Utilities to try-loading vocab.json and W.npy
// -----------------------------------------------------------------------------

fn try_load_vocab<P: AsRef<Path>>(p: P) -> Option<Arc<Vocab>> {
    match Vocab::load_from_json(p) {
        Ok(v) => Some(Arc::new(v)),
        Err(e) => {
            warn!("Failed to load vocab.json: {}", e);
            None
        }
    }
}

fn try_load_w<P: AsRef<Path>>(p: P) -> Option<Arc<Array2<f32>>> {
    match read_npy(p) {
        Ok(arr) => Some(Arc::new(arr)),
        Err(e) => {
            warn!("Failed to load W.npy: {}", e);
            None
        }
    }
}

fn try_load_labels<P: AsRef<Path>>(p: P) -> Option<Arc<Labels>> {
    match Labels::load_from_txt(p) {
        Ok(l) => Some(Arc::new(l)),
        Err(e) => {
            warn!("Failed to load labels file: {}", e);
            None
        }
    }
}


// -----------------------------------------------------------------------------
// WebSocketProxyServer: accepts clients and wires everything together.
// For simplicity we use one BatchManager per client connection in this baseline.
// You could move batch manager to be global and share across connections if you prefer.
// -----------------------------------------------------------------------------

pub struct WebSocketProxyServer {
    address: String,
    // we keep an Arc<NaiveBayes> for fallback classification
    nb: Arc<NaiveBayes>,
    // optional shared W matrix / vocab (loaded once and reused). use OnceCell to load once.
    shared_vocab: OnceCell<Arc<Vocab>>,
    shared_w: OnceCell<Option<Arc<Array2<f32>>>>,
}

impl WebSocketProxyServer {
    pub fn new(addr: &str, nb: Arc<NaiveBayes>) -> Self {
        Self {
            address: addr.to_string(),
            nb,
            shared_vocab: OnceCell::new(),
            shared_w: OnceCell::new(),
        }
    }

    /// Configure shared artifacts once (optional). You can call this before run().
    /// Example: server.load_shared_artifacts(Some("vocab.json"), Some("W.npy"));
    pub fn load_shared_artifacts<P: AsRef<Path>>(&self, vocab_path: Option<P>, w_path: Option<P>) {
        if let Some(v) = vocab_path {
            if let Ok(vocab) = Vocab::load_from_json(v) {
                let _ = self.shared_vocab.set(Arc::new(vocab));
                info!("Loaded vocab.json into shared cache");
            } else {
                warn!("Failed to load shared vocab.json");
            }
        }
        if let Some(p) = w_path {
            if let Ok(arr) = read_npy(p) {
                let _ = self.shared_w.set(Some(Arc::new(arr)));
                info!("Loaded W.npy into shared cache");
            } else {
                warn!("Failed to load shared W.npy");
                let _ = self.shared_w.set(None);
            }
        }


    }

    /// Handle a single TCP connection (WebSocket handshake done here)
    async fn handle_connection(&self, stream: TcpStream) {
        let config = Some(WebSocketConfig::default());
        let ws_stream = match accept_async_with_config(stream, config).await {
            Ok(ws) => ws,
            Err(e) => { error!("WebSocket accept error: {}", e); return; }
        };

        let (write, mut read) = ws_stream.split();
        let mut_write = Arc::new(Mutex::new(write));

        // Channel to send outgoing results to the client
        let (tx_ws, mut rx_ws) = mpsc::channel::<String>(256);

        // Channel for raw MessageItems to be batched
        let (batch_tx, mut batch_rx) = mpsc::channel::<MessageItem>(1024);

        // Build transformer: prefer shared vocab, else try to load default file, else None
        // If no vocab -> we will not use vectorized path (fallback to nb)
        let transformer: Arc<dyn Transformer> = if let Some(v) = self.shared_vocab.get() {
            Arc::new(NdarrayTransformer::new(v.clone()))
        } else if let Some(vocab) = try_load_vocab(VOCAB_PATH) {
            let _ = self.shared_vocab.set(vocab.clone());
            Arc::new(NdarrayTransformer::new(vocab))
        } else {
            // No vocab available: create a toy transformer that does nothing (counts 0)
            // But we keep the trait object to keep types consistent.
            let empty_vocab = Arc::new(Vocab::new_from_iter(Vec::<String>::new()));
            Arc::new(NdarrayTransformer::new(empty_vocab))
        };

        // Prepare pools: create typed channels (BatchMatrix) per pool
        // Try to load labels once (best-effort). Use labels.txt or whatever path you have.
        let maybe_labels: Option<Arc<Labels>> = try_load_labels(LABELS_PATH);

        let num_pools = 2_usize.max(1); // small default; tune later via config
        let mut pool_senders: Vec<mpsc::Sender<Arc<BatchMatrix>>> = Vec::with_capacity(num_pools);
        for i in 0..num_pools {
            let (tx_batchmat, rx_batchmat) = mpsc::channel::<Arc<BatchMatrix>>(8);
            pool_senders.push(tx_batchmat);

            // For worker, try to use shared W if available, else None
            let maybe_w: Option<Arc<Array2<f32>>> = match self.shared_w.get() {
                Some(opt) => opt.clone(),
                None => {
                    // try to load W.npy in current dir (best-effort)
                    try_load_w(W_PATH)
                }
            };

            // Worker uses vectorized path if W present, else falls back to NaiveBayes
            let nb_clone = Some(self.nb.clone());
            let tx_ws_clone = Some(tx_ws.clone());
            let worker = PoolWorker::new(i, rx_batchmat, maybe_w.clone(), nb_clone, tx_ws_clone, maybe_labels.clone());
            tokio::spawn(async move { worker.run().await });
        }

        // Dispatcher that takes BatchMatrix and routes to pool channels
        // We'll wrap pool_senders into a new Arc<Vec<Sender<BatchMatrix>>> and pass to Dispatcher.
        // Create an adapter that turns BatchMatrix into direct sends (we'll not use old trait)
        let dispatcher = Arc::new(Dispatcher::new(pool_senders));

        // Create BatchManager: buffer size and timeout ms - tuned for baseline
        let batch_manager = BatchManager::new(
            16,       // max batch size
            300,      // timeout ms
            transformer.clone(),
            dispatcher.clone(),
        );

        // Spawn consumer that reads MessageItem from batch_rx and feeds manager.
        // We keep manager in the consumer task to avoid mutexing it.
        let manager_handle = tokio::spawn(async move {
            // Note: we capture batch_manager by moving into the task
            let mut mgr = batch_manager;
            while let Some(msg) = batch_rx.recv().await {
                mgr.feed(msg).await;
            }
            // flush remaining
            mgr.flush().await;
        });

        // Spawn writer task that sends outgoing JSON strings to the client
        let writer_handle = tokio::spawn(async move {
            while let Some(payload) = rx_ws.recv().await {
                println!("{}", payload);
                let mut write = mut_write.lock().await;
                if write.send(Message::Text(payload)).await.is_err() {
                    // client gone
                    break;
                }
            }
        });

        // Create JetStream — we reuse its `handle_text` to parse incoming events and push eligible posts
        let url = "wss://jetstream.atproto.tools/subscribe";
        let jet = JetStream::new(url, batch_tx.clone());
        let jet = Arc::new(jet);
 
        let jet_copy = jet.clone();
        tokio::spawn(async move {
            if let Err(e) = jet_copy.run().await {
                error!("FeedCollectorWS failed: {}", e);
            }
        });

        // Read loop for client: we accept either control messages or event JSON to feed jet.
        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    println!("{}", text);
                    let _ = jet.handle_text(&text).await;
                }
                Ok(Message::Close(_)) => break,
                Err(e) => {
                    error!("Read loop error: {}", e);
                    break;
                }
                _ => {}
            }
        }

        // Client disconnected: drop batch_tx to end manager consumer
        drop(batch_tx);
        let _ = manager_handle.await;
        let _ = writer_handle.await;
        info!("Connection handler finished");
    }

    /// Start listening and accepting client connections
    pub async fn run(&self) -> Result<()> {
        info!("Resolving address: {}", self.address);
        let mut addrs = lookup_host(&self.address)
            .await
            .context("lookup_host failed")?;

        let addr = addrs.next().ok_or_else(|| anyhow::anyhow!("No addr found for host"))?;
        let listener = TcpListener::bind(&addr).await.context("bind failed")?;
        info!("Listening on {}", addr);

        loop {
            match listener.accept().await {
                Ok((stream, _peer)) => {
                    let server_clone = self.clone_for_task();
                    tokio::spawn(async move {
                        server_clone.handle_connection(stream).await;
                    });
                }
                Err(e) => {
                    error!("Accept failed: {}", e);
                }
            }
        }
    }

    /// Helper to clone the server for use inside a tokio task
    fn clone_for_task(&self) -> Self {
        Self {
            address: self.address.clone(),
            nb: self.nb.clone(),
            shared_vocab: self.shared_vocab.clone(),
            shared_w: self.shared_w.clone(),
        }
    }
}

// -----------------------------------------------------------------------------
// Implement Clone for JetStream (it only holds a Sender)
impl Clone for JetStream {
    fn clone(&self) -> Self {
        JetStream { url: self.url.clone(), batch_tx: self.batch_tx.clone() }
    }
}

// -----------------------------------------------------------------------------
// For WebSocketProxyServer: simple constructor + main entrypoint example
// -----------------------------------------------------------------------------

impl Clone for WebSocketProxyServer {
    fn clone(&self) -> Self {
        Self {
            address: self.address.clone(),
            nb: self.nb.clone(),
            shared_vocab: self.shared_vocab.clone(),
            shared_w: self.shared_w.clone(),
        }
    }
}

// -----------------------------------------------------------------------------
// Example main — run server at 127.0.0.1:9002
// -----------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing subscriber for logs
    tracing_subscriber::fmt::init();

    // Instantiate your NaiveBayes model (your file already provides constructors / load functions).
    // TODO: if your NaiveBayes has a loader (e.g. NaiveBayes::load(path)) use it here.
    let mut nb_model = NaiveBayes::load_from_file("model.json")?;
    nb_model.build_dense_matrix();
    let nb = Arc::new(nb_model);

    // Create server
    let server = WebSocketProxyServer::new("127.0.0.1:9002", nb);

    // Optionally preload vocab and W if you have them at known paths:
    server.load_shared_artifacts(Some("C:\\Users\\cepha\\OneDrive\\Bureau\\Cube\\gaello v.2\\rs_bluesky\\vocab.json"), Some("C:\\Users\\cepha\\OneDrive\\Bureau\\Cube\\gaello v.2\\rs_bluesky\\w.npy"));

    // Run server (this never returns unless error)
    server.run().await?;

    Ok(())
}
