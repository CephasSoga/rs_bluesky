

// jetstream.rs
//
// Rewritten to match intended architecture:
// - Global shared pipeline (not per-connection)
// - Central PoolManager with dynamic scaling
// - LogRouter for result distribution
// - Separation of WebSocket handling from processing pipeline
//
// Architecture:
// WebSocket/JetStream → GlobalPipeline → PoolManager → Workers → LogRouter → Clients/DB
//

#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(clippy::needless_pass_by_value)]


static GLOBAL_SHUTDOWN: AtomicBool = AtomicBool::new(false);
static ACTIVE_CLIENTS: AtomicUsize = AtomicUsize::new(0);

use std::sync::Arc;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::time::{Instant, Duration};

use tokio::sync::{Mutex, RwLock, mpsc, oneshot};
use tokio::sync::mpsc::{Sender, Receiver};
use tokio::{task, time};
use tokio::signal;
use tokio_stream::wrappers::ReceiverStream;
use futures_util::{StreamExt, SinkExt};
use async_tungstenite::tungstenite::protocol::Message;
use async_tungstenite::tungstenite::error::Error;
use async_tungstenite::tokio::{TokioAdapter, connect_async};
use async_tungstenite::tokio::accept_async_with_config;
use tungstenite::protocol::WebSocketConfig;
use tokio::net::{TcpListener, TcpStream, lookup_host};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{info, warn, error, debug};

use ndarray::{Array2, ArrayView2, Axis};
use ndarray_npy::read_npy;
use once_cell::sync::OnceCell;
use anyhow::{Result, Context};

// Configuration constants
const LABELS_PATH: &str = "C:\\Users\\cepha\\OneDrive\\Bureau\\Cube\\gaello v.2\\rs_bluesky\\labels.txt";
const VOCAB_PATH: &str = "C:\\Users\\cepha\\OneDrive\\Bureau\\Cube\\gaello v.2\\rs_bluesky\\vocab.json";
const W_PATH: &str = "C:\\Users\\cepha\\OneDrive\\Bureau\\Cube\\gaello v.2\\rs_bluesky\\w.npy";
const NB_PATH: &str = "C:\\Users\\cepha\\OneDrive\\Bureau\\Cube\\gaello v.2\\rs_bluesky\\naive_bayes_model.json";

use crate::classifiers::naive_bayes::NaiveBayes;
use crate::config::Config;
use crate::logging::setup_logger;

// -----------------------------------------------------------------------------
// Core Types
// -----------------------------------------------------------------------------

type ClientId = usize;

#[derive(Clone, Debug)]
pub struct MessageItem {
    pub id: usize,
    pub text: String,
    pub client_id: Option<ClientId>,
    pub timestamp: Instant,
    pub source: MessageSource,
}

#[derive(Clone, Debug)]
pub enum MessageSource {
    WebSocketClient(ClientId),
    JetStream,
    Internal,
}

#[derive(Clone)]
pub struct BatchMatrix {
    pub x: Array2<f32>,
    pub items: Vec<MessageItem>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ClassificationResult {
    pub message_id: usize,
    pub client_id: Option<ClientId>,
    pub text: String,
    pub label: String,
    pub confidence: f32,
    pub processing_time_ms: u64,
    pub timestamp: i64,
    pub source: String,
}

// -----------------------------------------------------------------------------
// Vocabulary and Labels (same as before)
// -----------------------------------------------------------------------------

#[derive(Clone)]
pub struct Vocab {
    pub token_to_idx: HashMap<String, usize>,
}

impl Vocab {
    pub fn new_from_iter<I: IntoIterator<Item = String>>(iter: I) -> Self {
        let mut token_to_idx = HashMap::new();
        for (i, tok) in iter.into_iter().enumerate() {
            token_to_idx.insert(tok, i);
        }
        Self { token_to_idx }
    }

    pub fn load_from_json<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        let json: Value = serde_json::from_slice(&bytes)?;
        let tokens: Vec<String> = match json {
            Value::Array(arr) => arr
                .into_iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
            Value::Object(obj) => obj.keys().cloned().collect(),
            _ => anyhow::bail!("Unsupported vocab format"),
        };
        Ok(Vocab::new_from_iter(tokens))
    }

    pub fn len(&self) -> usize { self.token_to_idx.len() }
}

pub struct Labels {
    pub vec: Vec<String>,
}

impl Labels {
    pub fn load_from_txt<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
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

    pub fn len(&self) -> usize { self.vec.len() }
}

// -----------------------------------------------------------------------------
// Transformer (same as before)
// -----------------------------------------------------------------------------

pub struct Tokenizer;

impl Tokenizer {
    pub fn tokenize<'a>(s: &'a str) -> impl Iterator<Item = String> + 'a {
        s.split_whitespace()
            .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|t| !t.is_empty())
            .map(|t| t.to_string())
    }
}

pub trait Transformer: Send + Sync {
    fn to_batch_matrix(&self, batch: &[MessageItem]) -> BatchMatrix;
}

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
        let mut x = Array2::<f32>::zeros((batch_size, vocab_size));

        for (i, msg) in batch.iter().enumerate() {
            for token in Tokenizer::tokenize(&msg.text) {
                if let Some(&idx) = self.vocab.token_to_idx.get(&token) {
                    x[[i, idx]] += 1.0;
                }
            }
        }

        BatchMatrix { x, items: batch.to_vec() }
    }
}

// -----------------------------------------------------------------------------
// Pool Management - Dynamic Scaling
// -----------------------------------------------------------------------------

#[derive(Clone)]
pub struct PoolConfig {
    pub min_pools: usize,
    pub max_pools: usize,
    pub scale_up_threshold: usize,    // avg queue depth to scale up
    pub scale_down_threshold: usize,  // avg queue depth to scale down
    pub queue_capacity: usize,
    pub scale_check_interval_ms: u64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            min_pools: 2,
            max_pools: 8,
            scale_up_threshold: 5,
            scale_down_threshold: 2,
            queue_capacity: 16,
            scale_check_interval_ms: 5000,
        }
    }
}

struct PoolInfo {
    id: usize,
    sender: mpsc::Sender<Arc<BatchMatrix>>,
    queue_size: Arc<AtomicUsize>,
    active_tasks: Arc<AtomicUsize>,
    shutdown: Arc<AtomicBool>,
}

pub struct PoolManager {
    pools: Arc<RwLock<Vec<PoolInfo>>>,
    config: PoolConfig,
    next_pool_id: AtomicUsize,
    log_router: Arc<LogRouter>,
    // Shared resources
    w_matrix: Option<Arc<Array2<f32>>>,
    nb_classifier: Option<Arc<NaiveBayes>>,
    labels: Option<Arc<Labels>>,
}

impl PoolManager {
    pub fn new(
        config: PoolConfig,
        log_router: Arc<LogRouter>,
        w_matrix: Option<Arc<Array2<f32>>>,
        nb_classifier: Option<Arc<NaiveBayes>>,
        labels: Option<Arc<Labels>>,
    ) -> Self {
        Self {
            pools: Arc::new(RwLock::new(Vec::new())),
            config,
            next_pool_id: AtomicUsize::new(0),
            log_router,
            w_matrix,
            nb_classifier,
            labels,
        }
    }

    pub async fn start(&self) -> Result<()> {
        // Initialize minimum pools
        for _ in 0..self.config.min_pools {
            self.spawn_pool().await;
        }

        // Start scaling monitor
        let manager = self.clone();
        tokio::spawn(async move {
            manager.scaling_monitor().await;
        });

        info!("PoolManager started with {} initial pools", self.config.min_pools);
        Ok(())
    }

    async fn spawn_pool(&self) {
        let pool_id = self.next_pool_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = mpsc::channel::<Arc<BatchMatrix>>(self.config.queue_capacity);
        let queue_size = Arc::new(AtomicUsize::new(0));
        let active_tasks = Arc::new(AtomicUsize::new(0));
        let shutdown = Arc::new(AtomicBool::new(false));

        let pool_info = PoolInfo {
            id: pool_id,
            sender: tx,
            queue_size: queue_size.clone(),
            active_tasks: active_tasks.clone(),
            shutdown: shutdown.clone(),
        };

        // Spawn worker
        let worker = PoolWorker::new(
            pool_id,
            rx,
            queue_size,
            active_tasks,
            shutdown,
            self.w_matrix.clone(),
            self.nb_classifier.clone(),
            self.labels.clone(),
            self.log_router.clone(),
        );

        tokio::spawn(async move {
            worker.run().await;
        });

        let mut pools = self.pools.write().await;
        pools.push(pool_info);
        
        info!("Spawned new pool worker {}", pool_id);
    }

    async fn remove_pool(&self) -> bool {
        let mut pools = self.pools.write().await;
        if pools.len() <= self.config.min_pools {
            return false;
        }

        if let Some(pool) = pools.pop() {
            pool.shutdown.store(true, Ordering::Relaxed);
            info!("Marked pool {} for shutdown", pool.id);
            return true;
        }
        false
    }

    async fn calculate_avg_queue_depth(&self) -> usize {
        let pools = self.pools.read().await;
        if pools.is_empty() {
            return 0;
        }

        let total: usize = pools.iter()
            .map(|p| p.queue_size.load(Ordering::Relaxed))
            .sum();
        total / pools.len()
    }

    async fn scaling_monitor(&self) {
        let mut interval = time::interval(Duration::from_millis(self.config.scale_check_interval_ms));
        
        loop {
            interval.tick().await;
            
            let avg_depth = self.calculate_avg_queue_depth().await;
            let pool_count = self.pools.read().await.len();

            if avg_depth > self.config.scale_up_threshold && pool_count < self.config.max_pools {
                info!("Scaling up: avg_depth={}, pools={}", avg_depth, pool_count);
                self.spawn_pool().await;
            } else if avg_depth < self.config.scale_down_threshold && pool_count > self.config.min_pools {
                info!("Scaling down: avg_depth={}, pools={}", avg_depth, pool_count);
                self.remove_pool().await;
            }
        }
    }

    pub async fn dispatch(&self, batch: Arc<BatchMatrix>) {
        let pools = self.pools.read().await;
        if pools.is_empty() {
            warn!("No pools available, dropping batch");
            return;
        }

        // Round-robin with fallback
        let start_idx = rand::random::<usize>() % pools.len();
        for i in 0..pools.len() {
            let idx = (start_idx + i) % pools.len();
            let pool = &pools[idx];

            if pool.shutdown.load(Ordering::Relaxed) {
                continue;
            }

            match pool.sender.try_send(batch.clone()) {
                Ok(_) => {
                    pool.queue_size.fetch_add(1, Ordering::Relaxed);
                    return;
                }
                Err(mpsc::error::TrySendError::Full(_)) => {
                    warn!("Pool {} queue full, trying next", pool.id);
                    continue;
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    warn!("Pool {} channel closed", pool.id);
                    continue;
                }
            }
        }

        warn!("All pools full or unavailable, dropping batch");
    }
}

impl Clone for PoolManager {
    fn clone(&self) -> Self {
        Self {
            pools: self.pools.clone(),
            config: self.config.clone(),
            next_pool_id: AtomicUsize::new(self.next_pool_id.load(Ordering::Relaxed)),
            log_router: self.log_router.clone(),
            w_matrix: self.w_matrix.clone(),
            nb_classifier: self.nb_classifier.clone(),
            labels: self.labels.clone(),
        }
    }
}

// -----------------------------------------------------------------------------
// Pool Worker - Enhanced with Metrics
// -----------------------------------------------------------------------------

pub struct PoolWorker {
    id: usize,
    rx: mpsc::Receiver<Arc<BatchMatrix>>,
    queue_size: Arc<AtomicUsize>,
    active_tasks: Arc<AtomicUsize>,
    shutdown: Arc<AtomicBool>,
    w_matrix: Option<Arc<Array2<f32>>>,
    nb_classifier: Option<Arc<NaiveBayes>>,
    labels: Option<Arc<Labels>>,
    log_router: Arc<LogRouter>,
}

impl PoolWorker {
    pub fn new(
        id: usize,
        rx: mpsc::Receiver<Arc<BatchMatrix>>,
        queue_size: Arc<AtomicUsize>,
        active_tasks: Arc<AtomicUsize>,
        shutdown: Arc<AtomicBool>,
        w_matrix: Option<Arc<Array2<f32>>>,
        nb_classifier: Option<Arc<NaiveBayes>>,
        labels: Option<Arc<Labels>>,
        log_router: Arc<LogRouter>,
    ) -> Self {
        Self {
            id,
            rx,
            queue_size,
            active_tasks,
            shutdown,
            w_matrix,
            nb_classifier,
            labels,
            log_router,
        }
    }

    pub async fn run(mut self) {
        info!("Pool worker {} started", self.id);
        
        while !self.shutdown.load(Ordering::Relaxed) {
            tokio::select! {
                batch = self.rx.recv() => {
                    match batch {
                        Some(batch) => {
                            self.queue_size.fetch_sub(1, Ordering::Relaxed);
                            self.active_tasks.fetch_add(1, Ordering::Relaxed);
                            
                            self.process_batch(batch).await;
                            
                            self.active_tasks.fetch_sub(1, Ordering::Relaxed);
                        }
                        None => {
                            debug!("Pool worker {} channel closed", self.id);
                            break;
                        }
                    }
                }
                _ = time::sleep(Duration::from_millis(100)) => {
                    // Check shutdown flag periodically
                    continue;
                }
            }
        }

        info!("Pool worker {} shutting down", self.id);
    }

    async fn process_naive_bayes(&self, batch: Arc<BatchMatrix>, nb: &NaiveBayes, start_time: Instant) {
        let router = self.log_router.clone();
        let nb = nb.clone();
        let pool_id = self.id;
        let batch_clone = batch.clone();

        task::spawn_blocking(move || {
            let texts: Vec<String> = batch_clone.items.iter().map(|item| item.text.clone()).collect();
            let results = nb.classify_batch_dense(&texts);

            for (i, label) in results.into_iter().enumerate() {
                let msg = &batch_clone.items[i];

                let result = ClassificationResult {
                    message_id: msg.id,
                    client_id: msg.client_id,
                    text: msg.text.clone(),
                    label,
                    confidence: 1.0, // NB doesn't provide confidence in this impl
                    processing_time_ms: start_time.elapsed().as_millis() as u64,
                    timestamp: chrono::Utc::now().timestamp(),
                    source: format!("nb_pool_{}", pool_id),
                };

                let router = router.clone();
                tokio::spawn(async move {
                    router.route_result(result).await;
                });
            }
        }).await.unwrap_or_else(|e| {
            error!("NaiveBayes processing failed in pool {}: {}", pool_id, e);
        });
    }

    async fn process_batch(&self, batch: Arc<BatchMatrix>) {
        let start_time = Instant::now();
        let batch_size = batch.items.len();

        info!("⚙️ Pool {}: Processing batch of {} items", self.id, batch_size);
        
        // Log each item in the batch
        for (i, item) in batch.items.iter().enumerate() {
            debug!("  - Item {}: message {} from client {:?}", i, item.id, item.client_id);
        }

        if let Some(w_matrix) = &self.w_matrix {
            info!("🔢 Pool {}: Using vectorized classifier", self.id);
            self.process_vectorized(batch, w_matrix, start_time).await;
        } else if let Some(nb) = &self.nb_classifier {
            info!("🧠 Pool {}: Using NaiveBayes classifier", self.id);
            self.process_naive_bayes(batch, nb, start_time).await;
        } else {
            error!("❌ Pool {}: No classifier available!", self.id);
        }

        info!("✅ Pool {}: Completed batch processing in {:?}", 
               self.id, start_time.elapsed());
    }

    async fn process_vectorized(&self, batch: Arc<BatchMatrix>, w_matrix: &Array2<f32>, start_time: Instant) {
        info!("🔢 Pool {}: Starting vectorized processing", self.id);
        
        let router = self.log_router.clone();
        let labels = self.labels.clone();
        let pool_id = self.id;
        let batch_clone = batch.clone();
        let w_matrix = w_matrix.to_owned();

        let processing_result = task::spawn_blocking(move || {
            info!("🧮 Pool {}: Computing scores matrix", pool_id);
            let scores: Array2<f32> = batch_clone.x.dot(&w_matrix.t());
            info!("📊 Pool {}: Scores matrix: {}x{}", pool_id, scores.nrows(), scores.ncols());

            let mut results = Vec::new();
            
            for (i, row) in scores.rows().into_iter().enumerate() {
                let mut best_idx = 0usize;
                let mut best_val = std::f32::NEG_INFINITY;
                
                for (j, v) in row.iter().enumerate() {
                    if *v > best_val {
                        best_val = *v;
                        best_idx = j;
                    }
                }

                let msg = &batch_clone.items[i];
                let label = labels
                    .as_ref()
                    .and_then(|l| l.get(best_idx))
                    .cloned()
                    .unwrap_or_else(|| format!("class_{}", best_idx));

                let result = ClassificationResult {
                    message_id: msg.id,
                    client_id: msg.client_id,
                    text: msg.text.clone(),
                    label: label.clone(),
                    confidence: best_val,
                    processing_time_ms: start_time.elapsed().as_millis() as u64,
                    timestamp: chrono::Utc::now().timestamp(),
                    source: format!("vectorized_pool_{}", pool_id),
                };

                info!("🏷️ Pool {}: Classified message {} -> {} (confidence: {:.3})", 
                      pool_id, msg.id, label, best_val);
                
                results.push(result);
            }
            
            results
        }).await;

        match processing_result {
            Ok(results) => {
                info!("✅ Pool {}: Processing complete, routing {} results", self.id, results.len());
                for result in results {
                    let router = router.clone();
                    tokio::spawn(async move {
                        router.route_result(result).await;
                    });
                }
            }
            Err(e) => {
                error!("❌ Pool {}: Vectorized processing failed: {}", self.id, e);
            }
        }
    }
}

// Test function to verify client connectivity
pub async fn test_client_connection(client_id: ClientId, pipeline: Arc<GlobalPipeline>) {
    info!("🧪 Testing client {} connectivity", client_id);
    
    // Send a test message
    pipeline.ingest_from_client(client_id, "Hello world test message".to_string()).await;
    
    // Wait a bit for processing
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    info!("🧪 Test message sent for client {}", client_id);
}

// -----------------------------------------------------------------------------
// Log Router - Central Result Distribution
// -----------------------------------------------------------------------------

pub struct LogRouter {
    // Channel for database persistence
    db_sender: Option<mpsc::Sender<ClassificationResult>>,
    // Channel for metrics collection
    metrics_sender: Option<mpsc::Sender<ClassificationResult>>,
    // WebSocket client senders
    client_senders: Arc<RwLock<HashMap<ClientId, mpsc::Sender<String>>>>,
}

impl LogRouter {
    pub fn new() -> Self {
        Self {
            db_sender: None,
            metrics_sender: None,
            client_senders: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn with_db_sender(mut self, sender: mpsc::Sender<ClassificationResult>) -> Self {
        self.db_sender = Some(sender);
        self
    }

    pub fn with_metrics_sender(mut self, sender: mpsc::Sender<ClassificationResult>) -> Self {
        self.metrics_sender = Some(sender);
        self
    }

    pub async fn register_client(&self, client_id: ClientId, sender: mpsc::Sender<String>) {
        let mut clients = self.client_senders.write().await;
        clients.insert(client_id, sender);
        info!("Registered client {}", client_id);

        // Update active clients count
        let n = ACTIVE_CLIENTS.fetch_add(1, Ordering::Relaxed) + 1;
        GLOBAL_SHUTDOWN.store(false, Ordering::Relaxed);
        info!("✅ Client registered, active_clients = {}", n);
    }

    pub async fn unregister_client(&self, client_id: ClientId) {
        let mut clients = self.client_senders.write().await;
        clients.remove(&client_id);
        info!("Unregistered client {}", client_id);

        // Update active clients count
        let n = ACTIVE_CLIENTS.fetch_sub(1, Ordering::Relaxed) - 1;
        info!("❌ Client unregistered, active_clients = {}", n);

        if n == 0 {
            info!("🛑 No active clients, setting shutdown flag");
            GLOBAL_SHUTDOWN.store(true, Ordering::Relaxed);
        }
    }

    pub async fn route_result(&self, result: ClassificationResult) {
        info!("🔄 LogRouter: Routing result for message {} from client {:?}", 
              result.message_id, result.client_id);

        // Send to database (fire and forget)
        if let Some(db_tx) = &self.db_sender {
            match db_tx.try_send(result.clone()) {
                Ok(_) => debug!("📊 DB: Sent result {}", result.message_id),
                Err(e) => warn!("📊 DB: Failed to send result {}: {}", result.message_id, e),
            }
        }

        // Send to metrics (fire and forget)
        if let Some(metrics_tx) = &self.metrics_sender {
            match metrics_tx.try_send(result.clone()) {
                Ok(_) => debug!("📈 Metrics: Sent result {}", result.message_id),
                Err(e) => warn!("📈 Metrics: Failed to send result {}: {}", result.message_id, e),
            }
        }

        // Send to specific client if applicable
        if let Some(client_id) = result.client_id {
            let clients = self.client_senders.read().await;
            info!("👥 LogRouter: {} active clients registered", clients.len());
            
            if let Some(client_tx) = clients.get(&client_id) {
                match serde_json::to_string(&result) {
                    Ok(payload) => {
                        info!("📤 LogRouter: Sending to client {}: {}", client_id, payload.chars().take(100).collect::<String>());
                        match client_tx.try_send(payload) {
                            Ok(_) => info!("✅ LogRouter: Successfully sent to client {}", client_id),
                            Err(mpsc::error::TrySendError::Full(_)) => {
                                warn!("⚠️ LogRouter: Client {} channel full", client_id);
                            }
                            Err(mpsc::error::TrySendError::Closed(_)) => {
                                warn!("❌ LogRouter: Client {} channel closed", client_id);
                            }
                        }
                    }
                    Err(e) => error!("🚫 LogRouter: Failed to serialize result: {}", e),
                }
            } else {
                warn!("❓ LogRouter: Client {} not found in registry", client_id);
                // List all registered clients for debugging
                for (id, _) in clients.iter() {
                    debug!("  - Registered client: {}", id);
                }
            }
        } else {
            debug!("🌐 LogRouter: Result has no client_id (from JetStream): {}", result.message_id);
        }

        // Always log for debugging
        info!("📝 Classified message {}: '{}' -> {} (confidence: {:.3})", 
              result.message_id, 
              result.text.chars().take(30).collect::<String>(), 
              result.label,
              result.confidence);
    }
}

// -----------------------------------------------------------------------------
// Global Pipeline - Core Orchestrator
// -----------------------------------------------------------------------------

pub struct GlobalPipeline {
    batch_manager: Arc<Mutex<BatchManager>>,
    pool_manager: Arc<PoolManager>,
    log_router: Arc<LogRouter>,
    transformer: Arc<dyn Transformer>,
    id_counter: AtomicUsize,
}

impl GlobalPipeline {
    pub async fn new(
        w_matrix: Option<Arc<Array2<f32>>>,
        nb_classifier: Option<Arc<NaiveBayes>>,
        vocab: Option<Arc<Vocab>>,
        labels: Option<Arc<Labels>>,
    ) -> Result<Self> {
        let log_router = Arc::new(LogRouter::new());
        let pool_config = PoolConfig::default();

        let pool_manager = Arc::new(PoolManager::new(
            pool_config,
            log_router.clone(),
            w_matrix,
            nb_classifier,
            labels,
        ));

        // Start pool manager
        pool_manager.start().await?;

        let transformer: Arc<dyn Transformer> = match vocab {
            Some(v) => Arc::new(NdarrayTransformer::new(v)),
            None => {
                let empty_vocab = Arc::new(Vocab::new_from_iter(Vec::<String>::new()));
                Arc::new(NdarrayTransformer::new(empty_vocab))
            }
        };

        let batch_manager = Arc::new(Mutex::new(BatchManager::new(
            32,  // batch size
            500, // timeout ms
            transformer.clone(),
            pool_manager.clone(),
        )));

        // Start batch manager consumer
        let bm_clone = batch_manager.clone();
        tokio::spawn(async move {
            let mut interval = time::interval(Duration::from_millis(100));
            loop {
                interval.tick().await;

                if GLOBAL_SHUTDOWN.load(Ordering::Relaxed) {
                    info!("🛑 BatchManager flush loop shutting down");
                    break;
                }


                let mut bm = bm_clone.lock().await;
                bm.check_timeout_flush().await;
            }
        });

        Ok(Self {
            batch_manager,
            pool_manager,
            log_router,
            transformer,
            id_counter: AtomicUsize::new(0),
        })
    }

    pub fn get_log_router(&self) -> Arc<LogRouter> {
        self.log_router.clone()
    }

    pub async fn ingest_from_client(&self, client_id: ClientId, text: String) {
        let msg = MessageItem {
            id: self.id_counter.fetch_add(1, Ordering::Relaxed),
            text,
            client_id: Some(client_id),
            timestamp: Instant::now(),
            source: MessageSource::WebSocketClient(client_id),
        };

        let mut bm = self.batch_manager.lock().await;
        bm.feed(msg).await;
    }

    pub async fn ingest_from_jetstream(&self, text: String) {
        // Early exit if shutting down
        if GLOBAL_SHUTDOWN.load(Ordering::Relaxed) || ACTIVE_CLIENTS.load(Ordering::Relaxed) == 0 {
            //info!("No active clients, skipping JetStream ingestion");
            return;
        }

        let cl_senders = self.pool_manager.log_router.client_senders.clone();

        let clients = cl_senders.read().await;            

        info!("🚁🚁🚁 {} active clients", ACTIVE_CLIENTS.load(Ordering::Relaxed));

        // collect all client IDs into a Vec
        let ids: Vec<ClientId> = clients.keys().cloned().collect();

        let mut msgs = Vec::with_capacity(ids.len());

        for id in ids {
            msgs.push(MessageItem {
                id: self.id_counter.fetch_add(1, Ordering::Relaxed),
                text: text.clone(),
                client_id: Some(id),
                timestamp: Instant::now(),
                source: MessageSource::JetStream,
            });
        }

        let mut bm = self.batch_manager.lock().await;
        for msg in msgs {
            bm.feed(msg).await;
        }
    }
}

// -----------------------------------------------------------------------------
// Enhanced Batch Manager
// -----------------------------------------------------------------------------

pub struct BatchManager {
    buffer: Vec<MessageItem>,
    max_size: usize,
    timeout_ms: u64,
    transformer: Arc<dyn Transformer>,
    pool_manager: Arc<PoolManager>,
    last_flush: Instant,
}

impl BatchManager {
    pub fn new(
        max_size: usize,
        timeout_ms: u64,
        transformer: Arc<dyn Transformer>,
        pool_manager: Arc<PoolManager>,
    ) -> Self {
        Self {
            buffer: Vec::with_capacity(max_size),
            max_size,
            timeout_ms,
            transformer,
            pool_manager,
            last_flush: Instant::now(),
        }
    }

    pub async fn feed(&mut self, item: MessageItem) {
        // Check shutdown before processing
        if GLOBAL_SHUTDOWN.load(Ordering::Relaxed) {
            debug!("BatchManager: Dropping message due to shutdown");
            return;
        }

        info!("🍽️ BatchManager: Feeding message {} from client {:?}: '{}'", 
              item.id, item.client_id, item.text.chars().take(50).collect::<String>());
        
        self.buffer.push(item);
        
        info!("📦 BatchManager: Buffer size: {}/{}", self.buffer.len(), self.max_size);
        
        if self.buffer.len() >= self.max_size {
            info!("🚀 BatchManager: Buffer full, flushing batch");
            self.flush().await;
        }
    }

    pub async fn check_timeout_flush(&mut self) {
        if self.timeout_ms > 0 && !self.buffer.is_empty() {
            let elapsed = self.last_flush.elapsed().as_millis() as u64;
            if elapsed >= self.timeout_ms {
                self.flush().await;
            }
        }
    }

    pub async fn flush(&mut self) {
        if self.buffer.is_empty() {
            debug!("📦 BatchManager: Buffer empty, skipping flush");
            return;
        }

        let batch_items = std::mem::take(&mut self.buffer);
        let batch_size = batch_items.len();
        
        info!("🔄 BatchManager: Flushing batch of {} items", batch_size);
        for item in &batch_items {
            debug!("  - Message {}: client {:?}", item.id, item.client_id);
        }
        
        let bm = self.transformer.to_batch_matrix(&batch_items);
        let bm = Arc::new(bm);
        
        info!("📊 BatchManager: Created batch matrix {}x{}", bm.x.nrows(), bm.x.ncols());
        
        self.pool_manager.dispatch(bm).await;
        self.last_flush = Instant::now();
        
        info!("✅ BatchManager: Batch dispatched successfully");
    }
}

// -----------------------------------------------------------------------------
// JetStream Integration
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

pub struct JetStream {
    url: String,
    pipeline: Arc<GlobalPipeline>,
}

impl JetStream {
    pub fn new(url: &str, pipeline: Arc<GlobalPipeline>) -> Self {
        Self {
            url: url.to_string(),
            pipeline,
        }
    }

    pub async fn run(&self) -> Result<()> {
        info!("Connecting to JetStream at {}", self.url);
        
        let (ws_stream, _) = connect_async(&self.url).await?;
        let (_, mut read) = ws_stream.split();

        while let Some(msg) = read.next().await {
            // Check shutdown flag in the main message loop
            if GLOBAL_SHUTDOWN.load(Ordering::Relaxed) {
                info!("JetStream shutting down due to global shutdown flag");
                break;
            }

            match msg {
                Ok(Message::Text(txt)) => {
                    // Only process if we have active clients
                    if ACTIVE_CLIENTS.load(Ordering::Relaxed) > 0 {
                        self.handle_text(&txt).await;
                    }
                }
                Ok(Message::Close(_)) => {
                    info!("JetStream connection closed");
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

    async fn handle_text(&self, text: &str) {
        // Early return if no active clients
        if ACTIVE_CLIENTS.load(Ordering::Relaxed) == 0 {
            return;
        }

        let evt: Event = match serde_json::from_str(text) {
            Ok(e) => e,
            Err(_) => return, // silently ignore parse errors
        };

        let commit = match evt.commit {
            Some(c) => c,
            None => return,
        };

        if (commit.operation == "create" || commit.operation == "update")
            && commit.collection == "app.bsky.feed.post"
        {
            if let Some(record) = commit.record {
                if let Ok(post) = serde_json::from_value::<FeedPost>(record) {
                    self.pipeline.ingest_from_jetstream(post.text).await;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// WebSocket Proxy Server - Simplified to Connection Management Only
// -----------------------------------------------------------------------------

pub struct WebSocketProxyServer {
    address: String,
    pipeline: Arc<GlobalPipeline>,
    client_counter: AtomicUsize,
}

impl WebSocketProxyServer {
    pub fn new(address: String, pipeline: Arc<GlobalPipeline>) -> Self {
        Self {
            address,
            pipeline,
            client_counter: AtomicUsize::new(0),
        }
    }

    pub async fn run(&self) -> Result<()> {
        info!("Resolving address: {}", self.address);
        let mut addrs = lookup_host(&self.address)
            .await
            .context("lookup_host failed")?;

        let addr = addrs.next().ok_or_else(|| anyhow::anyhow!("No addr found for host"))?;
        let listener = TcpListener::bind(&addr).await.context("bind failed")?;
        info!("WebSocket server listening on {}", addr);

        loop {
            info!("🎈🎈🔄 WebSocket server: Waiting for new connection...");
            match listener.accept().await {
                Ok((stream, peer)) => {
                    info!("🎈 New connection attempt from: {}", peer);
                    let client_id = self.client_counter.fetch_add(1, Ordering::Relaxed);
                    info!("🌟 New client connection: {} (client_id: {})", peer, client_id);
                    
                    let pipeline = self.pipeline.clone();
                    tokio::spawn(async move {
                        if let Err(e) = Self::handle_client(stream, client_id, pipeline).await {
                            error!("❌ Client {} error: {}", client_id, e);
                        }
                    });
                }
                Err(e) => {
                    error!("Accept failed: {}", e);
                }
            }
        }

    }

    async fn handle_client(
        stream: TcpStream,
        client_id: ClientId,
        pipeline: Arc<GlobalPipeline>,
    ) -> Result<()> {
            info!("🔌 WebSocket: Starting handler for client {}", client_id);
            
            let config = Some(WebSocketConfig::default());
            let ws_stream = accept_async_with_config(stream, config).await?;
            let (ws_sender, mut ws_receiver) = ws_stream.split();
            let ws_sender = Arc::new(Mutex::new(ws_sender));

            // Create channel for sending results back to this client
            let (result_tx, mut result_rx) = mpsc::channel::<String>(256);
            
            // Register client with log router
            let log_router = pipeline.get_log_router();
            log_router.register_client(client_id, result_tx).await;
            
            info!("✅ WebSocket: Client {} registered with LogRouter", client_id);

            let ws_sender_1 = ws_sender.clone();
            // Spawn task to send results back to client
            let client_writer = tokio::spawn(async move {
                info!("📡 WebSocket: Starting writer task for client {}", client_id);
                let mut msg_count = 0;
                
                while let Some(result) = result_rx.recv().await {
                    msg_count += 1;
                    info!("📨 WebSocket: Client {} writer received message #{}: {}", 
                        client_id, msg_count, result.chars().take(50).collect::<String>());
                    
                    let mut ws_sender = ws_sender_1.lock().await;
                    match ws_sender.send(Message::Text(result)).await {
                        Ok(_) => {
                            info!("✅ WebSocket: Successfully sent message #{} to client {}", msg_count, client_id);
                        }
                        Err(e) => {
                            error!("❌ WebSocket: Failed to send to client {}: {}", client_id, e);
                            break;
                        }
                    }
                }
                
                warn!("🔚 WebSocket: Writer task ended for client {} after {} messages", client_id, msg_count);
            });

            let ws_sender_2 = ws_sender.clone();
            let mut received_count = 0;
            
            // Main message processing loop
            while let Some(msg) = ws_receiver.next().await {
                match msg? {
                    Message::Text(text) => {
                        received_count += 1;
                        info!("📥 WebSocket: Client {} sent message #{}: {}", 
                            client_id, received_count, text.chars().take(100).collect::<String>());
                        
                        // Try to parse as JetStream event first, then treat as raw text
                        if text.starts_with("{") { // && text.contains("commit") {
                            info!("🌊 WebSocket: Processing JetStream format from client {}", client_id);
                            let temp_jetstream = JetStream::new("", pipeline.clone());
                            temp_jetstream.handle_text(&text).await;
                        } else {
                            info!("💬 WebSocket: Processing text message from client {}", client_id);
                            pipeline.ingest_from_client(client_id, text).await;
                        }
                    }
                    Message::Close(_) => {
                        info!("👋 WebSocket: Client {} sent close frame", client_id);
                        log_router.unregister_client(client_id).await;
                        break;
                    }
                    Message::Ping(data) => {
                        debug!("🏓 WebSocket: Client {} ping", client_id);
                        let mut ws_sender = ws_sender_2.lock().await;
                        ws_sender.send(Message::Pong(data)).await?;
                    }
                    _ => {
                        debug!("📦 WebSocket: Client {} sent other message type", client_id);
                    }
                }
            }

            // Cleanup
            info!("🧹 WebSocket: Cleaning up client {}", client_id);
            log_router.unregister_client(client_id).await;
            client_writer.abort();
            info!("👋 WebSocket: Client {} disconnected after {} messages", client_id, received_count);

            Ok(())
        }
    }

// -----------------------------------------------------------------------------
// Utility Functions
// -----------------------------------------------------------------------------

fn try_load_vocab<P: AsRef<Path>>(path: P) -> Option<Arc<Vocab>> {
    match Vocab::load_from_json(path) {
        Ok(v) => {
            info!("Loaded vocabulary from file");
            Some(Arc::new(v))
        }
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

fn try_load_labels<P: AsRef<Path>>(path: P) -> Option<Arc<Labels>> {
    match Labels::load_from_txt(path) {
        Ok(l) => {
            info!("Loaded {} labels from file", l.len());
            Some(Arc::new(l))
        }
        Err(e) => {
            warn!("Failed to load labels file: {}", e);
            None
        }
    }
}

// -----------------------------------------------------------------------------
// Database and Metrics Handlers (Stubs for Extension)
// -----------------------------------------------------------------------------

async fn database_handler(mut rx: mpsc::Receiver<ClassificationResult>) {
    info!("Database handler started");
    while let Some(result) = rx.recv().await {
        // TODO: Implement actual database persistence
        debug!("DB: Storing result for message {}", result.message_id);
        // Example: insert into database
        // db.insert_classification_result(result).await;
    }
    info!("Database handler stopped");
}

async fn metrics_handler(mut rx: mpsc::Receiver<ClassificationResult>) {
    info!("Metrics handler started");
    while let Some(result) = rx.recv().await {
        // TODO: Implement metrics collection
        debug!("METRICS: {} -> {} (confidence: {:.2})", 
               result.text.chars().take(30).collect::<String>(),
               result.label, 
               result.confidence);
        // Example: update Prometheus metrics, send to InfluxDB, etc.
    }
    info!("Metrics handler stopped");
}

// -----------------------------------------------------------------------------
// Main Application Entry Point
// -----------------------------------------------------------------------------

pub async fn exec(addr: &str) -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("Starting Global Pipeline Architecture");

    // Load shared resources
    let vocab = try_load_vocab(VOCAB_PATH);
    let w_matrix = try_load_w(W_PATH);
    let labels = try_load_labels(LABELS_PATH);

    // Load NaiveBayes classifier
    let nb_classifier = match NaiveBayes::load_from_file(NB_PATH) {
        Ok(mut nb) => {
            nb.build_dense_matrix();
            info!("Loaded NaiveBayes classifier");
            Some(Arc::new(nb))
        }
        Err(e) => {
            warn!("Failed to load NaiveBayes classifier: {}", e);
            None
        }
    };

    // Validate we have at least one classifier
    if w_matrix.is_none() && nb_classifier.is_none() {
        return Err(anyhow::anyhow!("No classifier available (neither W matrix nor NaiveBayes)"));
    }

    // Initialize global pipeline
    let pipeline = Arc::new(
        GlobalPipeline::new(w_matrix, nb_classifier, vocab, labels).await?
    );

    // Set up optional database and metrics handlers
    let log_router = pipeline.get_log_router();
    
    // Example: Enable database logging
    // let (db_tx, db_rx) = mpsc::channel(1000);
    // let enhanced_router = LogRouter::new().with_db_sender(db_tx);
    // tokio::spawn(database_handler(db_rx));

    // Example: Enable metrics collection
    // let (metrics_tx, metrics_rx) = mpsc::channel(1000);
    // let enhanced_router = enhanced_router.with_metrics_sender(metrics_tx);
    // tokio::spawn(metrics_handler(metrics_rx));

    // Start JetStream connection
    let jetstream = JetStream::new("wss://jetstream.atproto.tools/subscribe", pipeline.clone());
    let jetstream_handle = tokio::spawn(async move {
        loop {
            if let Err(e) = jetstream.run().await {
                error!("JetStream connection failed: {}", e);
                warn!("Reconnecting to JetStream in 5 seconds...");
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        }
    });

    // Start WebSocket server
    let server = WebSocketProxyServer::new(addr.to_string(), pipeline.clone());
    let server_handle = tokio::spawn(async move {
        if let Err(e) = server.run().await {
            error!("WebSocket server failed: {}", e);
        }
    });

     // Handle Ctrl+C gracefully
     // TODO: Add proper shutdown:
    // 1. Create Arc<AtomicBool> or Arc<Notify> as shutdown flag
    // 2. In ctrl_c handler -> set flag / notify
    // 3. Check flag in accept loop + client loops -> break gracefully
    // 4. Optionally broadcast close frames to clients
    // E.g: set flag 'static CTRL_C_SHUTDOWN: AtomicBool = AtomicBool::new(false);`
    // And use it accross all tasks to signal shutdown
    let shutdown_signal = async {
        signal::ctrl_c().await.expect("Failed to install CTRL+C signal handler");
        info!("Received CTRL+C, initiating graceful shutdown");
        GLOBAL_SHUTDOWN.store(true, Ordering::Relaxed);
    };

    info!("System fully initialized and running");
    info!("WebSocket server: ws://127.0.0.1:9002");
    info!("JetStream: Connected to Bluesky firehose");

    // Wait for either service to fail
    tokio::select! {
        result = jetstream_handle => {
            error!("JetStream task ended: {:?}", result);
        }
        result = server_handle => {
            error!("WebSocket server task ended: {:?}", result);
        }
        _ = shutdown_signal => {
            info!("Ctrl+C Shutdown signal received");
            // Wait for active clients to disconnect
            while ACTIVE_CLIENTS.load(Ordering::Relaxed) > 0 {
                info!("Waiting for {} active clients to disconnect...", ACTIVE_CLIENTS.load(Ordering::Relaxed));
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
            info!("All clients disconnected, shutting down");
        }
    }
    info!("Shutdown complete");

    Ok(())
}