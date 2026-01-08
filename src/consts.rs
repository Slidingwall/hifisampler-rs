pub const SAMPLE_RATE: u32 = 44100;
pub const FFT_SIZE: usize = 2048;
pub const HOP_SIZE: usize = 512;
pub const ORIGIN_HOP_SIZE: usize = 128;
pub const FEATURE_EXT: &str = "hifi.npz";
use ini::Ini;
use lazy_static::lazy_static;
use std::collections::HashMap;
use std::path::PathBuf;
#[derive(Debug, Clone, PartialEq)]
pub struct HifiConfig {
    pub vocoder_path: PathBuf,
    pub hnsep_path: PathBuf,
    pub wave_norm: bool,
    pub trim_silence: bool,
    pub silence_threshold: f64,
    pub loop_mode: bool,
    pub peak_limit: f64,
    pub fill: usize,
    pub max_workers: usize,
}
lazy_static! {
    pub static ref HIFI_CONFIG: HifiConfig = load_hifi_config();
}
fn load_hifi_config() -> HifiConfig {
    let ini = match Ini::load_from_file("hificonfig.ini") {
        Ok(ini) => ini,
        Err(_) => return HifiConfig::default(),
    };
    let default_section: HashMap<String, String> = ini
        .section(None::<String>)
        .map(|props| props.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect())
        .unwrap_or_default();
    HifiConfig {
        vocoder_path: default_section
            .get("vocoder_path")
            .cloned()
            .map(PathBuf::from)
            .unwrap_or(PathBuf::from("./model/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx")),
        hnsep_path: default_section
            .get("hnsep_path")
            .cloned()
            .map(PathBuf::from)
            .unwrap_or(PathBuf::from("./model/hnsep_model.onnx")),
        wave_norm: default_section
            .get("wave_norm")
            .and_then(|s| s.parse().ok())
            .unwrap_or(true),
        trim_silence: default_section
            .get("trim_silence")
            .and_then(|s| s.parse().ok())
            .unwrap_or(true),
        loop_mode: default_section
            .get("loop_mode")
            .and_then(|s| s.parse().ok())
            .unwrap_or(true),
        silence_threshold: default_section
            .get("silence_threshold")
            .and_then(|s| s.parse().ok())
            .unwrap_or(-52.0),
        peak_limit: default_section
            .get("peak_limit")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0),
        fill: default_section
            .get("fill")
            .and_then(|s| s.parse().ok())
            .unwrap_or(6),
        max_workers: default_section
            .get("max_workers")
            .and_then(|s| s.parse().ok())
            .unwrap_or(2),
    }
}
impl Default for HifiConfig {
    fn default() -> Self {
        Self {
            vocoder_path: PathBuf::from("./model/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx"),
            hnsep_path: PathBuf::from("./model/hnsep_model.onnx"),
            wave_norm: true,
            trim_silence: true,
            silence_threshold: -52.0,
            loop_mode: true,
            peak_limit: 1.0,
            fill: 6,
            max_workers: 2,
        }
    }
}
#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use super::*;
    #[test]
    fn test_default_config() {
        let default = HifiConfig::default();
        assert_eq!(
            default.vocoder_path,
            PathBuf::from("./model/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx")
        );
        assert_eq!(
            default.hnsep_path,
            PathBuf::from("./model/hnsep_model.onnx")
        );
        assert_eq!(default.wave_norm, true);
        assert_eq!(default.trim_silence, true);
        assert_eq!(default.silence_threshold, -52.0);
        assert_eq!(default.loop_mode, true);
        assert_eq!(default.peak_limit, 1.0);
        assert_eq!(default.fill, 6);
        assert_eq!(default.max_workers, 2);
    }
    #[test]
    fn test_global_config_init() {
        let cfg = &HIFI_CONFIG;
        assert!(!cfg.vocoder_path.as_os_str().is_empty());
        assert!(!cfg.hnsep_path.as_os_str().is_empty());
        assert!(cfg.silence_threshold.is_finite());
        assert!(cfg.peak_limit.is_finite());
        assert!(cfg.fill > 0);
        assert!(cfg.max_workers <= 32);
    }
    #[test]
    fn test_real_ini_load() {
        let ini_exists = Path::new("hificonfig.ini").exists();
        let cfg = &HIFI_CONFIG;
        if ini_exists {
            println!("Real hificonfig.ini exists, verify parsed result is valid");
            assert!(!cfg.vocoder_path.as_os_str().is_empty());
            assert!(!cfg.hnsep_path.as_os_str().is_empty());
        } else {
            println!("Real hificonfig.ini does not exist, verify default config is returned");
            assert_eq!(**cfg, HifiConfig::default());
        }
    }
    #[test]
    fn test_parse_fault_tolerance() {
        let cfg = &HIFI_CONFIG;
        assert!(cfg.silence_threshold.is_finite());
        assert!(cfg.peak_limit.is_finite());
        assert!(cfg.fill <= 100);
        assert!(cfg.max_workers >= 1 && cfg.max_workers <= 32);
    }
}