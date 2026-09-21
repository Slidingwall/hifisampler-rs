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
    let mut eps: Vec<ExecutionProviderDispatch> = Vec::new();
    #[cfg(feature = "ep-directml")]
    {
        let ep = ort::ep::DirectML::default();
        log_ep_availability(&ep);
        eps.push(ep.build());
    }
    #[cfg(feature = "ep-coreml")]
    {
        let ep = ort::ep::CoreML::default();
        log_ep_availability(&ep);
        eps.push(ep.build());
    }
    #[cfg(feature = "ep-webgpu")]
    {
        let ep = ort::ep::WebGPU::default();
        log_ep_availability(&ep);
        eps.push(ep.build());
    }
    #[cfg(feature = "ep-cuda")]
    {
        let trt = ort::ep::TensorRT::default();
        log_ep_availability(&trt);
        eps.push(trt.build());
        let cuda = ort::ep::CUDA::default();
        log_ep_availability(&cuda);
        eps.push(cuda.build());
    }
    eps.push(ort::ep::CPU::default().build());
    tracing::info!(
        "Execution provider chain (priority order): [{}]",
        eps.iter()
            .map(|e| format!("{:?}", e))
            .collect::<Vec<_>>()
            .join(" -> ")
    );
    eps
});
fn log_ep_availability<E: ort::ep::ExecutionProvider>(ep: &E) {
    let name = ep.name();
    match ep.is_available() {
        Ok(true) => tracing::info!("Execution provider '{name}' is compiled-in and AVAILABLE — GPU acceleration will be used."),
        Ok(false) => tracing::warn!("Execution provider '{name}' is compiled-in but NOT available on this host — falling back to CPU."),
        Err(e) => tracing::warn!("Could not query availability of EP '{name}' ({e}) — will fall back to CPU if registration fails."),
    }
}
fn select_execution_providers() -> Vec<ExecutionProviderDispatch> {
    EXECUTION_PROVIDERS.clone()
}
