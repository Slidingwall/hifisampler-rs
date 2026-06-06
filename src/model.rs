pub mod hnsep;
pub mod hifigan;
use std::sync::{Mutex, atomic::{AtomicUsize, Ordering}};
use once_cell::sync::OnceCell;
use crate::consts::HIFI_CONFIG;
use crate::model::{hifigan::HiFiGANLoader, hnsep::HNSEPLoader};
static VOCODER_POOL: OnceCell<Vec<Mutex<HiFiGANLoader>>> = OnceCell::new();
static REMOVER_POOL: OnceCell<Vec<Mutex<HNSEPLoader>>> = OnceCell::new();
static NEXT_VOCODER: AtomicUsize = AtomicUsize::new(0);
static NEXT_REMOVER: AtomicUsize = AtomicUsize::new(0);
pub fn initialize_models(max_workers: usize) {
    if !HIFI_CONFIG.vocoder_path.exists() {
        tracing::error!("HiFiGAN model not found at: {}", HIFI_CONFIG.vocoder_path.display());
    }
    if !HIFI_CONFIG.hnsep_path.exists() {
        tracing::error!("HNSEP model not found at: {}", HIFI_CONFIG.hnsep_path.display());
    }
    let cpu_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let pool_size = max_workers.min(cpu_cores).max(1);
    tracing::info!("Creating model pool with size = {}", pool_size);
    let vocoder_pool = (0..pool_size)
        .map(|_| Mutex::new(HiFiGANLoader::new(&HIFI_CONFIG.vocoder_path)))
        .collect();
    VOCODER_POOL.set(vocoder_pool).unwrap();
    let remover_pool = (0..pool_size)
        .map(|_| Mutex::new(HNSEPLoader::new(&HIFI_CONFIG.hnsep_path)))
        .collect();
    REMOVER_POOL.set(remover_pool).unwrap();
    tracing::info!("All models initialized successfully.");
}
pub fn get_vocoder() -> &'static Mutex<HiFiGANLoader> {
    let pool = VOCODER_POOL.get().expect("Vocoder pool not initialized");
    let idx = NEXT_VOCODER.fetch_add(1, Ordering::Relaxed) % pool.len();
    &pool[idx]
}
pub fn get_remover() -> &'static Mutex<HNSEPLoader> {
    let pool = REMOVER_POOL.get().expect("Remover pool not initialized");
    let idx = NEXT_REMOVER.fetch_add(1, Ordering::Relaxed) % pool.len();
    &pool[idx]
}