pub mod mel;
pub mod hnsep;
pub mod hifigan;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::Arc;
use crate::consts::HIFI_CONFIG;
use crate::model::{hifigan::HiFiGANLoader, hnsep::HNSEPLoader, mel::MelAnalyzer};
pub static VOCODER: OnceLock<Arc<Mutex<HiFiGANLoader>>> = OnceLock::new();
pub static REMOVER: OnceLock<Arc<Mutex<HNSEPLoader>>> = OnceLock::new();
pub static MEL_ANALYZER: OnceLock<Arc<MelAnalyzer>> = OnceLock::new();
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
    let melanalyzer = Arc::new(MelAnalyzer::new());
    MEL_ANALYZER.set(melanalyzer).unwrap();
    tracing::info!("MelAnalyzer initialized successfully");
    tracing::info!("All models initialized successfully.");
}
pub fn get_vocoder() -> Arc<Mutex<HiFiGANLoader>> {
    VOCODER.get().cloned().unwrap()
}
pub fn get_remover() -> Arc<Mutex<HNSEPLoader>> {
    REMOVER.get().cloned().unwrap()
}
pub fn get_mel_analyzer() -> Arc<MelAnalyzer> {
    MEL_ANALYZER.get().cloned().unwrap()
}