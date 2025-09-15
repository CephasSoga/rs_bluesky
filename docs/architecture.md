
---

# Entropy JetStream–Proxy Architecture

## 📖 Overview

This system ingests events from two main sources:

* **WebSocket clients** (user-facing connections)
* **JetStream feed** (external Bluesky-like firehose)

Events are validated, batched, transformed into numerical form, classified, and routed through pools of workers. All results and structured logs are aggregated via the **LogRouter**.

The design ensures:

* **Scalability**: pools and workers scale horizontally and vertically.
* **Resilience**: JetStream and WebSocket clients are decoupled from processing via batching and queues.
* **Observability**: labels and logs flow into a centralized router.

---

## 🏗 Architecture Diagram (code-aligned)

```
Incoming WebSocket Messages       JetStream Feed
        │                                │
        ▼                                ▼
  WebSocketProxyServer           JetStream::handle_text
        │                                │
        └───────────────┬────────────────┘
                        ▼
                  GlobalPipeline
                        │
                        ▼
                BatchManager (📦)
                        │
                        ▼
                  PoolManager (⚖️)
          ┌────────────┼─────────────┐
          ▼            ▼             ▼
        Pool_1       Pool_2       Pool_N
          │            │             │
   +-------------+ +-------------+ +-------------+
   | PoolWorker  | | PoolWorker  | | PoolWorker  |
   | Transformer | | Transformer | | Transformer |
   | Classifier  | | Classifier  | | Classifier  |
   +-------------+ +-------------+ +-------------+
          │            │             │
          └────────────┴─────────────┘
                        │
                        ▼
                    LogRouter
```

---

## ⚙️ Core Components

### `GlobalPipeline`

Defined in `global_pipeline.rs`

* Holds shared state:

  * `batch_manager: Arc<Mutex<BatchManager>>`
  * `pool_manager: Arc<PoolManager>`
  * `log_router: Arc<LogRouter>`
  * `id_counter: AtomicUsize`
* Provides ingestion methods:

  * `ingest_from_client(client_id, text)`
  * `ingest_from_jetstream(text)`

📌 **Reference:**

```rust
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
```

---

### `BatchManager`

Defined in `batch_manager.rs`

* Buffers messages until:

  * Max size reached (e.g. 32 items)
  * Timeout expires (e.g. 500 ms)
* Flushes batches into the `PoolManager`.

📌 **Reference:**

```rust
if self.buffer.len() >= self.max_size {
    info!("🚀 BatchManager: Buffer full, flushing batch");
    self.flush().await;
}
```

---

### `PoolManager`

Defined in `pool_manager.rs`

* Decides which pool gets a batch.
* Can **scale out** by creating new pools if needed.
* Maintains `client_senders: Arc<RwLock<HashMap<ClientId, Sender<String>>>>` for routing.

---

### `PoolWorker`

Defined in `pool_worker.rs`

* Runs in parallel within pools.
* For each batch:

  1. Transform text into numerical arrays (`MatrixTransformer`).
  2. Classify with either:

     * `W.npy` matrix (vectorized classification).
     * Naive Bayes fallback classifier.
  3. Emit labels + logs to `LogRouter`.

---

### `LogRouter`

Defined in `log_router.rs`

* Central aggregator for all labels and logs.
* Broadcasts structured logs back to clients.

---

## 🚀 Scaling Strategy

You can scale performance along three axes:

1. **Batch size / timeout**

   * Higher batch size → throughput ↑, latency ↑
   * Lower batch size → throughput ↓, latency ↓

2. **Workers per pool**

   * Add more async tasks (Tokio workers).
   * Vertical scaling (one pool, many workers).

3. **Number of pools**

   * PoolManager can spawn additional pools.
   * Horizontal scaling across pools.

📌 **Tip:** Start by scaling workers in a pool. If contention rises (lock on queue), scale out with new pools.

---

## 🛡 Shutdown and Lifecycle

* **Ctrl+C** handled via `tokio::signal::ctrl_c()`.
* Trigger a shutdown flag (`Arc<AtomicBool>` or `Arc<Notify>`).
* Components poll this flag to gracefully stop.
* Clients are **unregistered** on close frames:

  ```rust
  Message::Close(_) => {
      info!("👋 WebSocket: Client {} sent close frame", client_id);
      pipeline.get_log_router().unregister_client(client_id).await;
      break;
  }
  ```

---

## 📊 Telemetry & Monitoring

Suggested metrics to track:

* **BatchManager**:

  * Current buffer size
  * Average flush latency

* **PoolManager**:

  * Active pools count
  * Queue depth per pool

* **Workers**:

  * Processing time per batch
  * Throughput (messages/sec)

* **System-wide**:

  * Connected clients
  * Classification accuracy

---

## 📦 Summary

* **GlobalPipeline** ties together ingestion, batching, routing, and logging.
* **BatchManager** smooths incoming load into manageable chunks.
* **PoolManager** + **PoolWorkers** provide scalable compute.
* **LogRouter** closes the loop back to clients.
* Scaling = adjust **batch size**, **worker count**, **pool count**.

This modular design ensures **low latency for real-time events** while allowing throughput to grow with load.

---


![Design](https://imglink.io/i/90d5aed9-c482-48f0-adbf-8f1380f773ac.png)