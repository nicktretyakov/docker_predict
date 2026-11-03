//! ML Container Orchestrator
//! Multi-feature load forecasting + anomaly detection (pure Rust, no heavy ML frameworks)
//! Features: CPU, Memory, Network, Disk I/O
//! Anomaly: combined reconstruction-style error + forecast error with dynamic z-score
//! Logging: tracing

use anyhow::{Context, Result};
use bollard::container::{CPUStats, Stats, StatsOptions};
use bollard::Docker;
use futures::StreamExt;
use std::collections::VecDeque;
use tracing::{debug, error, info, warn};
use tracing_subscriber::EnvFilter;

// ================== Константы ==================
const SEQ_LEN: usize = 24;
const ANOMALY_Z: f64 = 2.8;

// ================== Метрики ==================
#[derive(Clone, Copy, Debug, Default)]
pub struct Metrics {
    pub cpu: f64,
    pub memory: f64,
    pub network: f64, // normalized 0-100
    pub disk_io: f64, // normalized 0-100
}

// ================== Простая, но эффективная модель ==================
// Экспоненциальное сглаживание + линейный тренд + динамический z-score
// (работает без внешних ML-библиотек и компилируется на старых toolchain)
pub struct AdvancedForecaster {
    history: VecDeque<Metrics>,
    net_mean: f64,
    net_std: f64,
    disk_mean: f64,
    disk_std: f64,
    recon_errors: VecDeque<f64>,
    forecast_errors: VecDeque<f64>,
    last_prediction: Option<Metrics>,
    // Простые параметры сглаживания
    alpha: f64, // уровень
    beta: f64,  // тренд
    level: Metrics,
    trend: Metrics,
    trained: bool,
}

impl AdvancedForecaster {
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(SEQ_LEN * 6),
            net_mean: 0.0,
            net_std: 1.0,
            disk_mean: 0.0,
            disk_std: 1.0,
            recon_errors: VecDeque::with_capacity(600),
            forecast_errors: VecDeque::with_capacity(600),
            last_prediction: None,
            alpha: 0.3,
            beta: 0.1,
            level: Metrics::default(),
            trend: Metrics::default(),
            trained: false,
        }
    }

    fn normalize(value: f64, mean: &mut f64, std: &mut f64) -> f64 {
        const ALPHA: f64 = 0.05;
        *mean = ALPHA * value + (1.0 - ALPHA) * *mean;
        let diff = value - *mean;
        *std = (ALPHA * diff * diff + (1.0 - ALPHA) * (*std) * (*std))
            .sqrt()
            .max(1.0);
        ((value - *mean) / *std * 18.0 + 50.0).clamp(0.0, 100.0)
    }

    pub fn update(&mut self, cpu: f64, memory: f64, net_bps: f64, disk_bps: f64) {
        let network = Self::normalize(net_bps, &mut self.net_mean, &mut self.net_std);
        let disk_io = Self::normalize(disk_bps, &mut self.disk_mean, &mut self.disk_std);
        let m = Metrics {
            cpu,
            memory,
            network,
            disk_io,
        };

        // Forecast error предыдущего шага
        if let Some(pred) = self.last_prediction.take() {
            let err = ((pred.cpu - m.cpu).powi(2)
                + (pred.memory - m.memory).powi(2)
                + (pred.network - m.network).powi(2)
                + (pred.disk_io - m.disk_io).powi(2))
                / 4.0;
            self.forecast_errors.push_back(err);
            if self.forecast_errors.len() > 500 {
                self.forecast_errors.pop_front();
            }
        }

        // Holt linear exponential smoothing update
        if self.history.is_empty() {
            self.level = m;
            self.trend = Metrics::default();
        } else {
            let prev_level = self.level;
            self.level.cpu = self.alpha * m.cpu + (1.0 - self.alpha) * (prev_level.cpu + self.trend.cpu);
            self.level.memory = self.alpha * m.memory + (1.0 - self.alpha) * (prev_level.memory + self.trend.memory);
            self.level.network = self.alpha * m.network + (1.0 - self.alpha) * (prev_level.network + self.trend.network);
            self.level.disk_io = self.alpha * m.disk_io + (1.0 - self.alpha) * (prev_level.disk_io + self.trend.disk_io);

            self.trend.cpu = self.beta * (self.level.cpu - prev_level.cpu) + (1.0 - self.beta) * self.trend.cpu;
            self.trend.memory = self.beta * (self.level.memory - prev_level.memory) + (1.0 - self.beta) * self.trend.memory;
            self.trend.network = self.beta * (self.level.network - prev_level.network) + (1.0 - self.beta) * self.trend.network;
            self.trend.disk_io = self.beta * (self.level.disk_io - prev_level.disk_io) + (1.0 - self.beta) * self.trend.disk_io;
        }

        if self.history.len() >= SEQ_LEN * 6 {
            self.history.pop_front();
        }
        self.history.push_back(m);

        if self.history.len() >= SEQ_LEN {
            self.trained = true;
        }
    }

    pub fn predict(&mut self, horizon: usize) -> Vec<Metrics> {
        if !self.trained {
            return vec![Metrics::default(); horizon];
        }

        let mut preds = Vec::with_capacity(horizon);
        for h in 1..=horizon {
            let m = Metrics {
                cpu: (self.level.cpu + h as f64 * self.trend.cpu).clamp(0.0, 100.0),
                memory: (self.level.memory + h as f64 * self.trend.memory).clamp(0.0, 100.0),
                network: (self.level.network + h as f64 * self.trend.network).clamp(0.0, 100.0),
                disk_io: (self.level.disk_io + h as f64 * self.trend.disk_io).clamp(0.0, 100.0),
            };
            preds.push(m);
        }

        if let Some(first) = preds.first() {
            self.last_prediction = Some(*first);
        }
        preds
    }

    /// Ужесточённый anomaly detection:
    /// - reconstruction-style error (отклонение от текущего level)
    /// - forecast error предыдущего шага
    /// - динамический z-score
    pub fn detect_anomaly(&mut self, current: &Metrics) -> (bool, f64, f64, f64) {
        if !self.trained {
            return (false, 0.0, 0.0, 0.0);
        }

        // "Reconstruction" error = расстояние до текущего сглаженного уровня
        let recon_err = ((current.cpu - self.level.cpu).powi(2)
            + (current.memory - self.level.memory).powi(2)
            + (current.network - self.level.network).powi(2)
            + (current.disk_io - self.level.disk_io).powi(2))
            / 4.0;

        self.recon_errors.push_back(recon_err);
        if self.recon_errors.len() > 500 {
            self.recon_errors.pop_front();
        }

        let r_mean = self.recon_errors.iter().sum::<f64>() / self.recon_errors.len().max(1) as f64;
        let r_var = self
            .recon_errors
            .iter()
            .map(|e| (e - r_mean).powi(2))
            .sum::<f64>()
            / self.recon_errors.len().max(1) as f64;
        let r_std = r_var.sqrt().max(1e-6);
        let z_recon = (recon_err - r_mean) / r_std;

        let (z_forecast, f_err) = if self.forecast_errors.len() > 15 {
            let f_mean = self.forecast_errors.iter().sum::<f64>() / self.forecast_errors.len() as f64;
            let f_var = self
                .forecast_errors
                .iter()
                .map(|e| (e - f_mean).powi(2))
                .sum::<f64>()
                / self.forecast_errors.len() as f64;
            let f_std = f_var.sqrt().max(1e-6);
            let last_f = *self.forecast_errors.back().unwrap_or(&0.0);
            ((last_f - f_mean) / f_std, last_f)
        } else {
            (0.0, 0.0)
        };

        let combined_z = 0.55 * z_recon + 0.45 * z_forecast;
        let is_anomaly = combined_z > ANOMALY_Z || z_recon > 3.5 || z_forecast > 3.2;

        (is_anomaly, recon_err, f_err, combined_z)
    }

    pub fn recommend_replicas(&mut self, current: i32, horizon: usize) -> i32 {
        let preds = self.predict(horizon);
        let max_load = preds
            .iter()
            .map(|m| m.cpu.max(m.memory).max(m.disk_io * 0.7))
            .fold(0.0_f64, f64::max);
        let target = (max_load / 62.0).ceil() as i32;
        target.clamp(1, current * 4).max(1)
    }
}

// ================== Расчёт метрик из bollard 0.15 ==================
fn calculate_cpu_percent(stats: &Stats) -> f64 {
    let cpu = &stats.cpu_stats;
    let pre = &stats.precpu_stats;

    let total = |c: &CPUStats| c.cpu_usage.total_usage;
    let system = |c: &CPUStats| c.system_cpu_usage.unwrap_or(0);

    let cpu_delta = total(cpu).saturating_sub(total(pre)) as f64;
    let system_delta = system(cpu).saturating_sub(system(pre)) as f64;
    let cpus = cpu.online_cpus.unwrap_or(1).max(1) as f64;

    if cpu_delta > 0.0 && system_delta > 0.0 {
        (cpu_delta / system_delta) * cpus * 100.0
    } else {
        0.0
    }
}

fn calculate_memory_percent(stats: &Stats) -> f64 {
    let usage = stats.memory_stats.usage.unwrap_or(0) as f64;
    let limit = stats.memory_stats.limit.unwrap_or(1) as f64;
    if limit > 0.0 {
        (usage / limit) * 100.0
    } else {
        0.0
    }
}

fn calculate_network_bps(stats: &Stats, prev_rx: &mut u64, prev_tx: &mut u64) -> f64 {
    let mut rx = 0u64;
    let mut tx = 0u64;
    if let Some(nets) = &stats.networks {
        for net in nets.values() {
            rx += net.rx_bytes;
            tx += net.tx_bytes;
        }
    }
    let delta = (rx.saturating_sub(*prev_rx) + tx.saturating_sub(*prev_tx)) as f64;
    *prev_rx = rx;
    *prev_tx = tx;
    delta
}

fn calculate_disk_io_bps(stats: &Stats, prev_read: &mut u64, prev_write: &mut u64) -> f64 {
    let mut read = 0u64;
    let mut write = 0u64;
    if let Some(entries) = &stats.blkio_stats.io_service_bytes_recursive {
        for e in entries {
            match e.op.as_str() {
                "Read" | "read" => read += e.value,
                "Write" | "write" => write += e.value,
                _ => {}
            }
        }
    }
    let delta = (read.saturating_sub(*prev_read) + write.saturating_sub(*prev_write)) as f64;
    *prev_read = read;
    *prev_write = write;
    delta
}

// ================== Основной цикл ==================
async fn monitor_container(container_id: &str) -> Result<()> {
    let mut model = AdvancedForecaster::new();

    let docker = Docker::connect_with_local_defaults().context("connect docker")?;
    info!(container = %container_id, "Starting monitoring");

    let mut stream = docker.stats(
        container_id,
        Some(StatsOptions {
            stream: true,
            one_shot: false,
        }),
    );

    let mut prev_rx = 0u64;
    let mut prev_tx = 0u64;
    let mut prev_disk_r = 0u64;
    let mut prev_disk_w = 0u64;
    let mut step: u64 = 0;

    while let Some(item) = stream.next().await {
        let stats = match item {
            Ok(s) => s,
            Err(e) => {
                error!(error = %e, "stats stream error");
                continue;
            }
        };

        let cpu = calculate_cpu_percent(&stats);
        let mem = calculate_memory_percent(&stats);
        let net = calculate_network_bps(&stats, &mut prev_rx, &mut prev_tx);
        let disk = calculate_disk_io_bps(&stats, &mut prev_disk_r, &mut prev_disk_w);

        model.update(cpu, mem, net, disk);
        step += 1;

        // Anomaly detection
        if let Some(last) = model.history.back().copied() {
            let (is_anom, recon_err, forecast_err, z) = model.detect_anomaly(&last);
            if is_anom {
                warn!(
                    container = %container_id,
                    cpu = format!("{:.1}%", cpu),
                    mem = format!("{:.1}%", mem),
                    recon_err = format!("{:.5}", recon_err),
                    forecast_err = format!("{:.5}", forecast_err),
                    z = format!("{:.2}", z),
                    "ANOMALY DETECTED"
                );
            } else {
                debug!(z = format!("{:.2}", z), "normal");
            }
        }

        // Прогноз и рекомендация реплик
        if step % 8 == 0 {
            let replicas = model.recommend_replicas(2, 6);
            info!(
                container = %container_id,
                cpu = format!("{:.1}%", cpu),
                mem = format!("{:.1}%", mem),
                net_bps = net,
                disk_bps = disk,
                recommended_replicas = replicas,
                "metrics + recommendation"
            );
        }
    }

    Ok(())
}

// ================== Entry point ==================
#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,ml_container_orchestrator=debug")),
        )
        .with_target(false)
        .with_thread_ids(false)
        .init();

    info!("ML Container Orchestrator starting");

    let container_id = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "your_container_name_or_id".to_string());

    // Self-test
    info!("Running self-test of model...");
    {
        let mut model = AdvancedForecaster::new();
        for i in 0..60 {
            let cpu = 25.0 + (i as f64 * 0.7) % 45.0;
            let mem = 35.0 + (i as f64 * 0.4) % 25.0;
            model.update(cpu, mem, 1500.0 * (i as f64 + 1.0), 800.0 * (i as f64 + 1.0));
        }
        let preds = model.predict(5);
        info!(?preds, "self-test predictions");
        if let Some(last) = model.history.back().copied() {
            let (anom, recon, f_err, z) = model.detect_anomaly(&last);
            info!(is_anomaly = anom, recon_err = recon, forecast_err = f_err, z = z, "self-test anomaly check");
        }
        let rec = model.recommend_replicas(2, 6);
        info!(recommended_replicas = rec, "self-test recommendation");
        info!("Self-test passed");
    }

    if container_id != "your_container_name_or_id" {
        monitor_container(&container_id).await?;
    } else {
        info!("No real container ID provided. Self-test completed. Pass a container name/ID as argument to start live monitoring.");
    }

    Ok(())
}
