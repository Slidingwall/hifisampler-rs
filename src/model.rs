pub mod hnsep;
pub mod hifigan;
use std::sync::{Arc, Mutex};
use once_cell::sync::OnceCell;
use crate::consts::HIFI_CONFIG;
use crate::model::{hifigan::HiFiGANLoader, hnsep::HNSEPLoader};
pub static VOCODER: OnceCell<Arc<Mutex<HiFiGANLoader>>> = OnceCell::new();
pub static REMOVER: OnceCell<Arc<Mutex<HNSEPLoader>>> = OnceCell::new();
pub fn initialize_models() {
    if !HIFI_CONFIG.vocoder_path.exists() {
        panic!("HiFiGAN model not found at: {}", HIFI_CONFIG.vocoder_path.display());
    }
    if !HIFI_CONFIG.hnsep_path.exists() {
        panic!("HNSEP model not found at: {}", HIFI_CONFIG.hnsep_path.display());
    }
    let hifigan = Arc::new(Mutex::new(HiFiGANLoader::new(&HIFI_CONFIG.vocoder_path)));
    VOCODER.set(hifigan).unwrap();
    tracing::info!("HiFiGAN model loaded successfully");
    let hnsep = Arc::new(Mutex::new(HNSEPLoader::new(&HIFI_CONFIG.hnsep_path)));
    REMOVER.set(hnsep).unwrap();
    tracing::info!("HNSEP model loaded successfully");
    tracing::info!("All models initialized successfully.");
}
pub fn get_vocoder() -> Arc<Mutex<HiFiGANLoader>> {
    VOCODER.get().cloned().unwrap()
}
pub fn get_remover() -> Arc<Mutex<HNSEPLoader>> {
    REMOVER.get().cloned().unwrap()
}