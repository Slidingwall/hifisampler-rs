mod hnsep;
mod hifigan;
use std::sync::{Mutex, atomic::{AtomicUsize, Ordering}};
use once_cell::sync::{Lazy, OnceCell};
use crate::consts::HIFI_CONFIG;
use crate::model::{hifigan::HiFiGANLoader, hnsep::HNSEPLoader};
use ort::ep::ExecutionProviderDispatch;
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
    let intra_threads = (cpu_cores / pool_size).max(1);
    tracing::info!("Creating model pool with size = {} (intra_threads = {})", pool_size, intra_threads);
    let vocoder_pool = (0..pool_size)
        .map(|_| Mutex::new(HiFiGANLoader::new(&HIFI_CONFIG.vocoder_path, intra_threads)))
        .collect();
    VOCODER_POOL.set(vocoder_pool).unwrap();
    let remover_pool = (0..pool_size)
        .map(|_| Mutex::new(HNSEPLoader::new(&HIFI_CONFIG.hnsep_path, intra_threads)))
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
static EXECUTION_PROVIDERS: Lazy<Vec<ExecutionProviderDispatch>> = Lazy::new(|| {
    #[cfg(target_os = "linux")]
    let eps = vec![ort::ep::CUDA::default().build(), ort::ep::WebGPU::default().build()];
    #[cfg(target_os = "windows")]
    let eps = vec![ort::ep::CUDA::default().build(), ort::ep::DirectML::default().build()];
    #[cfg(target_os = "macos")]
    let eps = vec![ort::ep::CoreML::default().build()];
    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    let eps: Vec<ExecutionProviderDispatch> = Vec::new();
    eps
});
fn select_execution_providers() -> Vec<ExecutionProviderDispatch> {
    EXECUTION_PROVIDERS.clone()
}
