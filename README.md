# ML Container Orchestrator

Lightweight container load forecasting and anomaly detection service written in pure Rust.

The service connects to a Docker daemon, continuously collects runtime metrics (CPU, memory, network, disk I/O), maintains an online forecasting model, detects anomalous behaviour, and emits replica-count recommendations.

---

## Features

| Capability | Description |
|---|---|
| Multi-metric ingestion | CPU %, Memory %, Network throughput, Disk I/O |
| Online forecasting | Holt linear exponential smoothing (level + trend) |
| Anomaly detection | Combined reconstruction error + forecast error with dynamic z-score |
| Replica recommendation | Capacity-aware suggestion derived from the forecast horizon |
| Structured logging | `tracing` with configurable log levels |
| Zero heavy ML deps | No Candle / PyTorch / ONNX runtime required |
| Built-in self-test | Synthetic data path for CI and smoke checks |

---

## Requirements

- Rust 1.70 or newer (1.80+ recommended)
- Docker Engine (for live monitoring)
- Linux, macOS, or Windows (WSL2)

---

## Quick Start

```bash
# Build
cargo build --release

# Smoke test (no Docker required)
cargo run --release

# Monitor a running container
cargo run --release -- <container_id_or_name>
```

Examples:

```bash
cargo run --release -- nginx
cargo run --release -- 7f3a9c2e1b84
```

### Log level

```bash
RUST_LOG=info,ml_container_orchestrator=debug \
  cargo run --release -- my-container
```

Supported levels: `error`, `warn`, `info`, `debug`, `trace`.

---

## Usage

### Typical workflow

1. Build the binary:
   ```bash
   cargo build --release
   ```

2. Verify the target container is running:
   ```bash
   docker ps
   ```

3. Start the orchestrator:
   ```bash
   ./target/release/ml-container-orchestrator my-app
   ```

4. Observe the log stream:
   - `metrics + recommendation` — current utilisation and suggested replica count
   - `ANOMALY DETECTED` — statistically significant deviation

5. Stop with `Ctrl+C`.

### Interpreting log messages

| Message | Meaning |
|---|---|
| `metrics + recommendation … recommended_replicas=N` | Current metrics and scaling suggestion |
| `ANOMALY DETECTED … z=X.XX` | Anomaly flagged (higher z → stronger deviation) |
| `Self-test passed` | Built-in model check succeeded |
| `stats stream error` | Failed to read Docker stats (container stopped or insufficient permissions) |

### Acting on recommendations

The service currently **emits** recommendations only. Scaling actions are left to the operator or an external controller:

```bash
# Docker Swarm
docker service scale my-service=4

# Kubernetes
kubectl scale deployment/my-app --replicas=4
```

### Permissions

The process needs permission to call `docker stats`. Either run as a user in the `docker` group or elevate:

```bash
sudo usermod -aG docker $USER   # then re-login
# or
sudo ./target/release/ml-container-orchestrator my-container
```

### Running as a service

**nohup**

```bash
nohup ./target/release/ml-container-orchestrator my-app \
  > orchestrator.log 2>&1 &
```

**systemd** (`/etc/systemd/system/ml-orchestrator.service`)

```ini
[Unit]
Description=ML Container Orchestrator
After=docker.service
Requires=docker.service

[Service]
Type=simple
ExecStart=/opt/ml-container-orchestrator/ml-container-orchestrator my-app
Restart=on-failure
RestartSec=5
Environment=RUST_LOG=info

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ml-orchestrator
```

---

## Testing

### Built-in self-test

```bash
cargo run --release
```

Validates:

- synthetic metric generation
- model update path
- multi-step forecast
- anomaly scoring
- replica recommendation

Expected output contains `Self-test passed`.

### Compilation checks

```bash
cargo check
cargo build --release
cargo clippy          # optional
```

### Live container test

```bash
docker run -d --name ml-test nginx:alpine
cargo run --release -- ml-test

# Optional load generation
docker exec ml-test sh -c "while true; do wget -q -O /dev/null http://localhost; done"
```

Stop with `Ctrl+C`, then clean up:

```bash
docker rm -f ml-test
```

### Unit-test sketch

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forecast_produces_horizon() {
        let mut model = AdvancedForecaster::new();
        for i in 0..40 {
            model.update(30.0 + i as f64 * 0.5, 40.0, 1000.0, 500.0);
        }
        let preds = model.predict(3);
        assert_eq!(preds.len(), 3);
        assert!(preds[0].cpu > 0.0);
    }
}
```

```bash
cargo test
```

---

## Architecture

```
Docker Stats API
       │
       ▼
┌──────────────────┐
│  Metric extract  │  CPU / Mem / Net / Disk
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ AdvancedForecaster│
│  • Holt smoothing │
│  • Forecast       │
│  • Anomaly score  │
│  • Replica advice │
└────────┬─────────┘
         │
         ▼
   Structured logs (tracing)
```

### Forecasting (Holt linear method)

```
levelₜ  = α · yₜ + (1 − α) · (levelₜ₋₁ + trendₜ₋₁)
trendₜ  = β · (levelₜ − levelₜ₋₁) + (1 − β) · trendₜ₋₁
ŷₜ₊ₕ   = levelₜ + h · trendₜ
```

Default parameters: `α = 0.3`, `β = 0.1`.

### Anomaly detection

1. Reconstruction error — distance of the current point from the smoothed level  
2. Forecast error — MSE of the previous prediction versus reality  
3. Dynamic z-score over both error streams  

An anomaly is raised when:

- `combined_z > 2.8`, or  
- `z_recon > 3.5`, or  
- `z_forecast > 3.2`

### Replica recommendation

```
max_load = max(CPU, Memory, Disk × 0.7) over the forecast horizon
replicas = clamp(ceil(max_load / 62), 1, current × 4)
```

Target utilisation band ≈ 60–65 %.

---

## Project layout

```
ml-container-orchestrator/
├── Cargo.toml
├── README.md
├── models/                 # reserved for future model snapshots
└── src/
    └── main.rs             # metrics, forecaster, monitoring loop
```

Key types and functions in `src/main.rs`:

| Symbol | Role |
|---|---|
| `Metrics` | Single observation (CPU, Mem, Net, Disk) |
| `AdvancedForecaster` | Online model (update / predict / detect / recommend) |
| `calculate_*` | Extraction helpers for bollard `Stats` |
| `monitor_container` | Async ingestion loop |
| `main` | Entry point + self-test |

---

## Configuration knobs

```rust
// src/main.rs
const SEQ_LEN: usize = 24;      // minimum history before forecasting
const ANOMALY_Z: f64 = 2.8;     // combined z-score threshold
```

Inside `AdvancedForecaster::new()`:

```rust
alpha: 0.3,   // level responsiveness
beta: 0.1,    // trend responsiveness
```

---

## Limitations

- Recommendations are advisory; no direct Docker Swarm / Kubernetes scale calls yet  
- Holt smoothing is effective for smooth or slowly trending loads; complex seasonality is better handled by LSTM-class models  
- Network and Disk normalisation is intentionally simple  

---

## Roadmap

- [ ] Optional LSTM / attention backend (Candle)  
- [ ] Native Kubernetes Deployment scaling  
- [ ] HTTP / WebSocket control plane  
- [ ] Model state persistence  
- [ ] Prometheus metrics exporter  

---

## License

MIT
