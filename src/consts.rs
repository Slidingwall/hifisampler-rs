pub const SAMPLE_RATE: u32 = 44100;
pub const FFT_SIZE: usize = 2048;
pub const HOP_SIZE: usize = 512;
pub const ORIGIN_HOP_SIZE: usize = 128;
pub const EPSILON: f32 = 1e-9;
pub const MEL_CENTER_HZ: [f32; 128] = [
    58.0953789, 76.6332474, 95.6244278, 115.080002, 135.011337, 155.430038, 176.348053, 197.777588,
    219.73114, 242.221512, 265.261871, 288.865631, 313.04657, 337.818817, 363.196808, 389.195404,
    415.829712, 443.115356, 471.068207, 499.70459, 529.041199, 559.095215, 589.884155, 621.425964,
    653.739075, 686.842346, 720.755127, 755.497131, 791.088745, 827.550659, 864.904175, 903.171082,
    942.373779, 982.535095, 1023.67853, 1065.828, 1109.00818, 1153.24426, 1198.56201, 1244.98792,
    1292.54919, 1341.27344, 1391.18909, 1442.32544, 1494.71216, 1548.37988, 1603.35999, 1659.68457,
    1717.38647, 1776.49927, 1837.05774, 1899.09692, 1962.6532, 2027.76367, 2094.46631, 2162.7998,
    2232.80444, 2304.521, 2377.99121, 2453.25806, 2530.36523, 2609.35815, 2690.28271, 2773.18579,
    2858.11646, 2945.12378, 3034.25879, 3125.57349, 3219.12109, 3314.9563, 3413.13477, 3513.71411,
    3616.75293, 3722.31152, 3830.45117, 3941.23535, 4054.72852, 4170.99707, 4290.1084, 4412.13281,
    4537.14062, 4665.20557, 4796.40234, 4930.80664, 5068.49805, 5209.55664, 5354.06396, 5502.10547,
    5653.76709, 5809.13672, 5968.30615, 6131.36768, 6298.4165, 6469.55029, 6644.86914, 6824.47461,
    7008.47217, 7196.96924, 7390.07568, 7587.90381, 7790.56982, 7998.19141, 8210.89062, 8428.79004,
    8652.01855, 8880.70508, 9114.98438, 9354.99219, 9600.86914, 9852.75879, 10110.8076, 10375.166,
    10645.9902, 10923.4355, 11207.666, 11498.8467, 11797.1484, 12102.7432, 12415.8115, 12736.5352,
    13065.1016, 13401.7031, 13746.5352, 14099.7988, 14461.7021, 14832.4541, 15212.2725, 15601.3789,
];
use ini::Ini;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::path::PathBuf;
#[derive(Debug, Clone, PartialEq)]
pub struct HifiConfig {
    pub vocoder_path: PathBuf,
    pub hnsep_path: PathBuf,
    pub wave_norm: bool,
    pub trim_silence: bool,
    pub silence_threshold: f32,
    pub loop_mode: bool,
    pub peak_limit: f32,
    pub fill: usize,
    pub max_workers: usize,
}
pub static HIFI_CONFIG: Lazy<HifiConfig> = Lazy::new(|| load_hifi_config());
fn load_hifi_config() -> HifiConfig {
    let ini = match Ini::load_from_file("hificonfig.ini") {
        Ok(ini) => ini,
        Err(_) => return HifiConfig::default(),
    };
    let def_sec: HashMap<String, String> = ini.section(None::<String>)
        .map(|props| props.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect())
        .unwrap_or_default();
    HifiConfig {
        vocoder_path: def_sec.get("vocoder_path").cloned().map(PathBuf::from)
            .unwrap_or(PathBuf::from("./model/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx")),
        hnsep_path: def_sec.get("hnsep_path").cloned().map(PathBuf::from)
            .unwrap_or(PathBuf::from("./model/hnsep_model.onnx")),
        wave_norm: def_sec.get("wave_norm").and_then(|s| s.parse().ok())
            .unwrap_or(true),
        trim_silence: def_sec.get("trim_silence").and_then(|s| s.parse().ok())
            .unwrap_or(true),
        loop_mode: def_sec.get("loop_mode").and_then(|s| s.parse().ok())
            .unwrap_or(true),
        silence_threshold: def_sec.get("silence_threshold").and_then(|s| s.parse().ok())
            .unwrap_or(-52.0),
        peak_limit: def_sec.get("peak_limit").and_then(|s| s.parse().ok())
            .unwrap_or(1.0),
        fill: def_sec.get("fill").and_then(|s| s.parse().ok())
            .unwrap_or(6),
        max_workers: def_sec.get("max_workers").and_then(|s| s.parse().ok())
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