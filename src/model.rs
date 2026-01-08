pub mod mel;
pub mod hnsep;
pub mod hifigan;
use std::sync::Mutex;
use std::sync::OnceLock; 
use std::sync::Arc;
use anyhow::{anyhow, Result};
use crate::consts::HIFI_CONFIG;
use crate::model::{hifigan::HiFiGANLoader, hnsep::HNSEPLoader, mel::MelAnalyzer};
pub static VOCODER: OnceLock<Arc<Mutex<HiFiGANLoader>>> = OnceLock::new();
pub static REMOVER: OnceLock<Arc<Mutex<HNSEPLoader>>> = OnceLock::new();
pub static MEL_ANALYZER: OnceLock<Arc<MelAnalyzer>> = OnceLock::new();
pub fn initialize_models() -> Result<(), anyhow::Error> {
    if !HIFI_CONFIG.vocoder_path.exists() {
        return Err(anyhow!(
            "HiFiGAN model not found at: {}",
            HIFI_CONFIG.vocoder_path.display()
        ));
    }
    if !HIFI_CONFIG.hnsep_path.exists() {
        return Err(anyhow!(
            "HNSEP model not found at: {}",
            HIFI_CONFIG.hnsep_path.display()
        ));
    }
    let hifigan = Arc::new(Mutex::new(HiFiGANLoader::new(&HIFI_CONFIG.vocoder_path)?));
    VOCODER.set(hifigan)
        .map_err(|_| anyhow!("HiFiGAN model already initialized"))?;
    tracing::info!("HiFiGAN model loaded successfully");
    let hnsep = Arc::new(Mutex::new(HNSEPLoader::new(&HIFI_CONFIG.hnsep_path)?));
    REMOVER.set(hnsep)
        .map_err(|_| anyhow!("HNSEP model already initialized"))?;
    tracing::info!("HNSEP model loaded successfully");
    let melanalyzer = Arc::new(MelAnalyzer::new());
    MEL_ANALYZER.set(melanalyzer)
        .map_err(|_| anyhow!("MelAnalyzer already initialized"))?;
    tracing::info!("MelAnalyzer initialized successfully");
    tracing::info!("All models initialized successfully.");
    Ok(())
}
pub fn get_vocoder() -> Result<Arc<Mutex<HiFiGANLoader>>> {
    VOCODER.get()
        .cloned()
        .ok_or_else(|| anyhow!("HiFiGAN model not initialized"))
}
pub fn get_remover() -> Result<Arc<Mutex<HNSEPLoader>>> {
    REMOVER.get()
        .cloned()
        .ok_or_else(|| anyhow!("HNSEP model not initialized"))
}
pub fn get_mel_analyzer() -> Result<Arc<MelAnalyzer>> {
    MEL_ANALYZER.get()
        .cloned()
        .ok_or_else(|| anyhow!("MelAnalyzer not initialized"))
}